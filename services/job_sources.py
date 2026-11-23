"""岗位来源适配器：只接"官方/公开 API"，不抓网页。

为什么这么克制：
    1. 招聘网站的 HTML 抓取既脆弱又容易违反条款；这里只调用招聘方自己
       对外提供的接口（Greenhouse / Lever / Ashby 的公开招聘板 API，
       RemoteOK / Arbeitnow 的公开 JSON API）。
    2. 来源可控才能谈"权威"：每个适配器都写死自己的域名，返回的链接
       还必须落在该域名的白名单里，否则丢弃——避免被接口里的脏数据带到
       钓鱼站点。
    3. 不做用户自定义 URL 抓取：那等于把服务端变成 SSRF 跳板。

安全约束（全部硬编码，不可由请求参数覆盖）：
    - 只允许 HTTPS；
    - 单次响应上限 3 MB、超时 12 秒；
    - 每个来源最小请求间隔，避免把对方打挂；
    - 只解析 JSON，不执行 JS、不带 cookie/凭据。
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse

MAX_BYTES = 3_000_000
TIMEOUT_SECONDS = 12.0
MIN_INTERVAL_SECONDS = 1.5
USER_AGENT = "CareerPilot/0.1 (self-hosted job search; +https://github.com/hylv123/careerpilot-mcp-agent)"

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")

_last_call: dict[str, float] = {}


class SourceError(RuntimeError):
    """来源不可用（网络、协议、格式）。"""


@dataclass(frozen=True)
class JobSource:
    id: str
    name: str
    kind: str                 # official_ats | public_api
    authority: str            # 给人看的权威性说明
    homepage: str
    link_hosts: tuple[str, ...]
    needs_board: bool
    build_url: Callable[[str | None], str]
    parse: Callable[[Any], list[dict[str, Any]]] = field(repr=False)


def _strip_html(text: Any, limit: int = 4000) -> str:
    cleaned = _TAG.sub(" ", str(text or ""))
    return _WS.sub(" ", cleaned).strip()[:limit]


def _host_allowed(url: str, hosts: tuple[str, ...]) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    host = parsed.hostname.casefold()
    return any(host == allowed or host.endswith("." + allowed) for allowed in hosts)


def http_get_json(url: str, *, source_id: str) -> Any:
    """带节流与大小限制的 JSON GET。"""
    if not url.startswith("https://"):
        raise SourceError(f"只允许 HTTPS：{url}")

    now = time.monotonic()
    last = _last_call.get(source_id, 0.0)
    if now - last < MIN_INTERVAL_SECONDS:
        time.sleep(MIN_INTERVAL_SECONDS - (now - last))
    _last_call[source_id] = time.monotonic()

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            if response.status != 200:
                raise SourceError(f"{source_id} 返回 HTTP {response.status}")
            raw = response.read(MAX_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SourceError(f"{source_id} 请求失败：{exc}") from exc

    if len(raw) > MAX_BYTES:
        raise SourceError(f"{source_id} 响应过大（>{MAX_BYTES} 字节）")
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise SourceError(f"{source_id} 返回的不是合法 JSON：{exc}") from exc


def _greenhouse(url_board: str | None) -> str:
    if not url_board:
        raise SourceError("Greenhouse 需要公司 board slug，例如 anthropic")
    return f"https://boards-api.greenhouse.io/v1/boards/{url_board}/jobs?content=true"


def _parse_greenhouse(payload: Any) -> list[dict[str, Any]]:
    jobs = payload.get("jobs", []) if isinstance(payload, dict) else []
    return [
        {
            "external_id": str(job.get("id", "")),
            "title": str(job.get("title", "")).strip(),
            "company": str((job.get("company_name") or "")).strip(),
            "location": str((job.get("location") or {}).get("name", "")).strip(),
            "url": str(job.get("absolute_url", "")).strip(),
            "description": _strip_html(job.get("content", "")),
            "posted_at": str(job.get("updated_at", "")).strip(),
        }
        for job in jobs
        if job.get("absolute_url")
    ]


def _lever(board: str | None) -> str:
    if not board:
        raise SourceError("Lever 需要公司 board slug")
    return f"https://api.lever.co/v0/postings/{board}?mode=json"


def _parse_lever(payload: Any) -> list[dict[str, Any]]:
    jobs = payload if isinstance(payload, list) else []
    rows = []
    for job in jobs:
        categories = job.get("categories") or {}
        rows.append(
            {
                "external_id": str(job.get("id", "")),
                "title": str(job.get("text", "")).strip(),
                "company": "",  # Lever 不返回公司名，由调用方用 board slug 兜底
                "location": str(categories.get("location", "")).strip(),
                "url": str(job.get("hostedUrl", "")).strip(),
                "description": _strip_html(job.get("descriptionPlain") or job.get("description", "")),
                "posted_at": str(job.get("createdAt", "")).strip(),
            }
        )
    return rows


def _ashby(board: str | None) -> str:
    if not board:
        raise SourceError("Ashby 需要公司 board slug")
    return f"https://api.ashbyhq.com/posting-api/job-board/{board}"


def _parse_ashby(payload: Any) -> list[dict[str, Any]]:
    jobs = payload.get("jobs", []) if isinstance(payload, dict) else []
    return [
        {
            "external_id": str(job.get("id", "")),
            "title": str(job.get("title", "")).strip(),
            "company": str(job.get("organizationName", "")).strip(),
            "location": str(job.get("location", "")).strip(),
            "url": str(job.get("jobUrl", "")).strip(),
            "description": _strip_html(job.get("descriptionHtml", "")),
            "posted_at": str(job.get("publishedAt", "")).strip(),
        }
        for job in jobs
        if job.get("jobUrl")
    ]


def _remoteok(_board: str | None) -> str:
    return "https://remoteok.com/api"


def _parse_remoteok(payload: Any) -> list[dict[str, Any]]:
    rows = []
    items = payload if isinstance(payload, list) else []
    for job in items:
        if not isinstance(job, dict) or not job.get("position"):
            continue  # 第一条是使用条款说明
        rows.append(
            {
                "external_id": str(job.get("id", "")),
                "title": str(job.get("position", "")).strip(),
                "company": str(job.get("company", "")).strip(),
                "location": str(job.get("location", "Remote")).strip() or "Remote",
                "url": str(job.get("url") or job.get("apply_url") or "").strip(),
                "description": _strip_html(job.get("description", "")),
                "posted_at": str(job.get("date", "")).strip(),
            }
        )
    return rows


def _arbeitnow(_board: str | None) -> str:
    return "https://www.arbeitnow.com/api/job-board-api"


def _parse_arbeitnow(payload: Any) -> list[dict[str, Any]]:
    items = payload.get("data", []) if isinstance(payload, dict) else []
    return [
        {
            "external_id": str(job.get("slug", "")),
            "title": str(job.get("title", "")).strip(),
            "company": str(job.get("company_name", "")).strip(),
            "location": str(job.get("location", "")).strip(),
            "url": str(job.get("url", "")).strip(),
            "description": _strip_html(job.get("description", "")),
            "posted_at": str(job.get("created_at", "")).strip(),
        }
        for job in items
        if job.get("url")
    ]


SOURCES: dict[str, JobSource] = {
    "greenhouse": JobSource(
        id="greenhouse",
        name="Greenhouse 招聘板",
        kind="official_ats",
        authority="招聘方官方 ATS 公开接口，链接即企业招聘页",
        homepage="https://boards-api.greenhouse.io/",
        link_hosts=("greenhouse.io",),
        needs_board=True,
        build_url=_greenhouse,
        parse=_parse_greenhouse,
    ),
    "lever": JobSource(
        id="lever",
        name="Lever 招聘板",
        kind="official_ats",
        authority="招聘方官方 ATS 公开接口，链接即企业招聘页",
        homepage="https://api.lever.co/",
        link_hosts=("lever.co",),
        needs_board=True,
        build_url=_lever,
        parse=_parse_lever,
    ),
    "ashby": JobSource(
        id="ashby",
        name="Ashby 招聘板",
        kind="official_ats",
        authority="招聘方官方 ATS 公开接口，链接即企业招聘页",
        homepage="https://api.ashbyhq.com/",
        link_hosts=("ashbyhq.com",),
        needs_board=True,
        build_url=_ashby,
        parse=_parse_ashby,
    ),
    "remoteok": JobSource(
        id="remoteok",
        name="RemoteOK",
        kind="public_api",
        authority="公开 JSON API，远程岗位聚合站",
        homepage="https://remoteok.com/api",
        link_hosts=("remoteok.com", "remoteok.io"),
        needs_board=False,
        build_url=_remoteok,
        parse=_parse_remoteok,
    ),
    "arbeitnow": JobSource(
        id="arbeitnow",
        name="Arbeitnow",
        kind="public_api",
        authority="公开 Job Board API，欧洲岗位为主",
        homepage="https://www.arbeitnow.com/api/job-board-api",
        link_hosts=("arbeitnow.com",),
        needs_board=False,
        build_url=_arbeitnow,
        parse=_parse_arbeitnow,
    ),
}


def list_sources() -> list[dict[str, Any]]:
    return [
        {
            "id": source.id,
            "name": source.name,
            "kind": source.kind,
            "authority": source.authority,
            "homepage": source.homepage,
            "needs_board": source.needs_board,
        }
        for source in SOURCES.values()
    ]


def fetch_source(source_id: str, *, board: str | None = None, limit: int = 30) -> list[dict[str, Any]]:
    """抓一个来源并归一化；链接不在白名单的记录直接丢弃。"""
    source = SOURCES.get(source_id)
    if source is None:
        raise SourceError(f"未知来源：{source_id}")

    payload = http_get_json(source.build_url(board), source_id=source_id)
    rows = source.parse(payload)

    cleaned: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        if not row.get("title") or not _host_allowed(row.get("url", ""), source.link_hosts):
            dropped += 1
            continue
        row.update({"source": source_id, "source_name": source.name})
        cleaned.append(row)
    if dropped:
        from loguru import logger

        logger.warning("[discovery] {} 丢弃 {} 条（缺标题或链接域名不在白名单）", source_id, dropped)
    return cleaned[: max(1, limit)]
