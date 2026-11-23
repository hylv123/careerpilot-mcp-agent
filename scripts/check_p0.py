"""P0 成果自检脚本。

用法（在 careerpilot 目录下，先启动 API）：

    .\.venv\Scripts\python.exe scripts/check_p0.py
    .\.venv\Scripts\python.exe scripts/check_p0.py --base-url http://127.0.0.1:8000
    .\.venv\Scripts\python.exe scripts/check_p0.py --write     # 额外验证 新建→更新 回环（会新增一条记录）
    .\.venv\Scripts\python.exe scripts/check_p0.py --analyze   # 额外跑一次真实 JD 分析（会消耗模型额度）

它只做只读检查，除非显式加 --write / --analyze。
默认检查项覆盖：
  1. /health 可用
  2. OpenAPI 已注册岗位接口
  3. /jobs 列表可读
  4. 状态过滤真的生效（返回记录的 status 与筛选一致）
  5. 非法状态返回 422
  6. 不存在的岗位返回 404（GET / PATCH / DELETE 都是）
  7. 每条记录都带 JD 概括描述，且长度受控
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Callable

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
STATUSES = ("未投", "已投", "笔试", "面试", "offer", "拒绝", "已归档")
MISSING_ID = 999_999_999

Check = tuple[str, Callable[[], str]]


def _get(base: str, path: str, **kwargs: Any) -> requests.Response:
    return requests.get(f"{base}{path}", timeout=10, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="CareerPilot P0 自检")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--write", action="store_true", help="验证新建→更新回环（会写入一条记录）")
    parser.add_argument("--analyze", action="store_true", help="验证一次真实 JD 分析（消耗额度）")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn: Callable[[], str]) -> None:
        try:
            results.append((name, True, fn()))
        except AssertionError as exc:
            results.append((name, False, str(exc)))
        except requests.RequestException as exc:
            results.append((name, False, f"请求失败：{exc}"))

    def health() -> str:
        resp = _get(base, "/health")
        assert resp.status_code == 200, f"HTTP {resp.status_code}"
        assert resp.json().get("status") == "ok", resp.text[:120]
        return "ok"

    def openapi() -> str:
        resp = _get(base, "/openapi.json")
        assert resp.status_code == 200, f"HTTP {resp.status_code}"
        paths = resp.json().get("paths", {})
        for path in ("/jobs", "/jobs/{job_id}"):
            assert path in paths, f"缺少 {path}"
        return "包含 /jobs 与 /jobs/{job_id}"

    jobs_holder: dict[str, list[dict[str, Any]]] = {}

    def list_jobs() -> str:
        resp = _get(base, "/jobs")
        assert resp.status_code == 200, f"HTTP {resp.status_code}"
        body = resp.json()
        assert isinstance(body, list), "返回值不是列表"
        jobs_holder["all"] = body
        return f"{len(body)} 条记录"

    def filter_status() -> str:
        sample = next((job["status"] for job in jobs_holder.get("all", [])), STATUSES[0])
        resp = _get(base, "/jobs", params={"status": sample})
        assert resp.status_code == 200, f"HTTP {resp.status_code}"
        body = resp.json()
        wrong = [job["id"] for job in body if job["status"] != sample]
        assert not wrong, f"筛选 {sample} 却返回了其它状态：{wrong[:3]}"
        return f"status={sample} → {len(body)} 条，全部匹配"

    def invalid_status() -> str:
        resp = _get(base, "/jobs", params={"status": "不存在的状态"})
        assert resp.status_code == 422, f"期望 422，实际 {resp.status_code}"
        return "422"

    def missing_job() -> str:
        got = _get(base, f"/jobs/{MISSING_ID}")
        assert got.status_code == 404, f"GET 期望 404，实际 {got.status_code}"
        patched = requests.patch(f"{base}/jobs/{MISSING_ID}", json={"notes": "probe"}, timeout=10)
        assert patched.status_code == 404, f"PATCH 期望 404，实际 {patched.status_code}"
        deleted = requests.delete(f"{base}/jobs/{MISSING_ID}", timeout=10)
        assert deleted.status_code == 404, f"DELETE 期望 404，实际 {deleted.status_code}"
        return "GET/PATCH/DELETE 均 404"

    def summary() -> str:
        jobs = jobs_holder.get("all", [])
        assert jobs, "没有岗位记录，无法检查 JD 概括（可先跑一次分析或加 --write）"
        missing = [job["id"] for job in jobs if not str(job.get("jd_summary") or "").strip()]
        assert not missing, f"这些记录缺少 jd_summary：{missing[:5]}"
        longest = max(len(job["jd_summary"]) for job in jobs)
        assert longest <= 141, f"概括过长：{longest} 字符"
        return f"{len(jobs)} 条都有概括，最长 {longest} 字符"

    check("GET /health", health)
    check("OpenAPI 注册岗位接口", openapi)
    check("GET /jobs 列表", list_jobs)
    check("状态过滤生效", filter_status)
    check("非法状态被拒", invalid_status)
    check("不存在岗位返回 404", missing_job)
    check("每条记录都有 JD 概括", summary)

    if args.write:
        holder: dict[str, Any] = {}

        def create() -> str:
            resp = requests.post(
                f"{base}/jobs",
                json={
                    "title": "自检临时记录",
                    "company": "自检",
                    "jd": "这是一条由 scripts/check_p0.py 生成的临时岗位记录，用于验证写入链路。",
                },
                timeout=10,
            )
            assert resp.status_code == 201, f"HTTP {resp.status_code}: {resp.text[:120]}"
            job = resp.json()
            assert job["status"] == "未投", f"默认状态应为 未投，实际 {job['status']}"
            holder["id"] = job["id"]
            return f"id={job['id']}"

        def update() -> str:
            resp = requests.patch(
                f"{base}/jobs/{holder['id']}",
                json={"status": "已归档", "notes": "自检后归档"},
                timeout=10,
            )
            assert resp.status_code == 200, f"HTTP {resp.status_code}"
            job = resp.json()
            assert job["status"] == "已归档", f"状态未更新：{job['status']}"
            assert job["notes"] == "自检后归档", "备注未更新"
            return "状态与备注已更新"

        def reject_bad_status() -> str:
            resp = requests.patch(f"{base}/jobs/{holder['id']}", json={"status": "乱写"}, timeout=10)
            assert resp.status_code == 422, f"期望 422，实际 {resp.status_code}"
            return "422"

        def delete() -> str:
            resp = requests.delete(f"{base}/jobs/{holder['id']}", timeout=10)
            assert resp.status_code == 204, f"期望 204，实际 {resp.status_code}"
            gone = _get(base, f"/jobs/{holder['id']}")
            assert gone.status_code == 404, f"删除后仍可读取：HTTP {gone.status_code}"
            again = requests.delete(f"{base}/jobs/{holder['id']}", timeout=10)
            assert again.status_code == 404, f"重复删除期望 404，实际 {again.status_code}"
            return "204，删除后 404，重复删除 404"

        check("POST /jobs 新建", create)
        check("PATCH /jobs 更新", update)
        check("PATCH 非法状态被拒", reject_bad_status)
        check("DELETE /jobs 删除", delete)

    if args.analyze:
        def analyze() -> str:
            resp = requests.post(
                f"{base}/analyze/sync",
                json={"jd": "招聘 AI Agent 开发实习生，要求熟悉 Python、LangChain、RAG，有 MCP 项目经验优先。"},
                timeout=600,
            )
            assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:160]}"
            body = resp.json()
            for key in ("analysis", "resume_bullets", "interview_questions", "match_assessment"):
                assert key in body, f"响应缺少 {key}"
            score = body["match_assessment"].get("score")
            assert isinstance(score, int), "match_assessment.score 不是整数"
            return f"覆盖分={score}，要点 {len(body['resume_bullets'])} 条，面试题 {len(body['interview_questions'])} 道"

        check("同步分析产出可解释匹配", analyze)

    print()
    width = max(len(name) for name, _, _ in results)
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {detail}")
    failed = [name for name, ok, _ in results if not ok]
    print()
    if failed:
        print(f"结果：{len(results) - len(failed)}/{len(results)} 通过，失败项：{', '.join(failed)}")
        return 1
    print(f"结果：{len(results)}/{len(results)} 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
