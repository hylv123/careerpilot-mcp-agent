"""候选人资料管理：读写 knowledge/docs/private 下的材料，并管理版本与索引。

为什么资料要单独一层，而不是让用户直接改 Markdown 文件：
    改完文件不等于检索生效——直读模式读的是 FAISS docstore（见
    knowledge/direct.py），不重建索引就还是旧内容。这一层把
    "保存 → 留版本 → 重建索引 → 清进程内缓存" 串成一次操作，
    避免再出现"资料改了、产出还是旧事实"的静默不一致。

存储约定：
    - 正文：knowledge/docs/private/<name>.md（参与 RAG 检索）
    - 历史版本：data/material_versions/<name>/<时间戳>.md
    - 索引：knowledge/index/（重建时由 build_index.py 覆盖）

不做元数据数据库：资料的名称、大小、更新时间全部从文件系统推导，
少一份状态就少一处会不一致的地方。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATERIALS_DIR = PROJECT_ROOT / "knowledge" / "docs" / "private"
VERSIONS_DIR = PROJECT_ROOT / "data" / "material_versions"
INDEX_DIR = PROJECT_ROOT / "knowledge" / "index"
BUILD_SCRIPT = PROJECT_ROOT / "knowledge" / "build_index.py"
LOCK_FILE = PROJECT_ROOT / "data" / ".reindex.lock"

MAX_CONTENT_CHARS = 200_000
MAX_NAME_CHARS = 60
_UNSAFE = re.compile(r"[^\w\u4e00-\u9fff .-]+")
_INDEX_FILES = ("index.faiss", "index.pkl")
LOCK_TTL_SECONDS = 600


class ProfileError(ValueError):
    """资料层的可预期错误（名称非法、资料不存在、内容为空等）。"""


class ReindexBusy(RuntimeError):
    """已有重建任务在跑。"""


def _materials_dir() -> Path:
    """允许测试通过环境变量指向临时目录。"""
    override = os.getenv("CAREERPILOT_MATERIALS_DIR", "").strip()
    return Path(override).resolve() if override else MATERIALS_DIR


def _versions_dir() -> Path:
    override = os.getenv("CAREERPILOT_VERSIONS_DIR", "").strip()
    return Path(override).resolve() if override else VERSIONS_DIR


def safe_name(name: str) -> str:
    """把用户输入变成安全的文件名主干，并挡掉路径穿越。"""
    raw = str(name or "").strip()
    if raw.lower().endswith(".md"):
        raw = raw[:-3]
    cleaned = _UNSAFE.sub("_", raw).strip(" ._")
    if not cleaned:
        raise ProfileError("资料名不能为空")
    if len(cleaned) > MAX_NAME_CHARS:
        raise ProfileError(f"资料名太长（最多 {MAX_NAME_CHARS} 字）")
    if cleaned in {".", ".."}:
        raise ProfileError("资料名非法")
    return cleaned


def _path_for(name: str) -> Path:
    return _materials_dir() / f"{safe_name(name)}.md"


def _require_content(content: str) -> str:
    text = str(content or "")
    if not text.strip():
        raise ProfileError("内容不能为空")
    if len(text) > MAX_CONTENT_CHARS:
        raise ProfileError(f"内容过长（最多 {MAX_CONTENT_CHARS} 字符）")
    return text


def _version_dir(name: str) -> Path:
    return _versions_dir() / safe_name(name)


def _snapshot(name: str, content: str, note: str = "") -> str | None:
    """把即将被覆盖的内容存一份历史版本，返回版本号。"""
    if not content.strip():
        return None
    directory = _version_dir(name)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    target = directory / f"{stamp}.md"
    target.write_text(content, encoding="utf-8")
    if note:
        (directory / f"{stamp}.note").write_text(str(note)[:200], encoding="utf-8")
    return stamp


def _index_mtime() -> float | None:
    times = [(INDEX_DIR / name).stat().st_mtime for name in _INDEX_FILES if (INDEX_DIR / name).exists()]
    return min(times) if len(times) == len(_INDEX_FILES) else None


def _newest_material_mtime() -> float | None:
    files = list(_materials_dir().glob("*.md"))
    return max((path.stat().st_mtime for path in files), default=None)


def _material_meta(path: Path) -> dict[str, Any]:
    index_mtime = _index_mtime()
    mtime = path.stat().st_mtime
    return {
        "name": path.stem,
        "filename": path.name,
        "size": path.stat().st_size,
        "updated_at": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "versions": len(list(_version_dir(path.stem).glob("*.md"))) if _version_dir(path.stem).exists() else 0,
        "indexed": bool(index_mtime and mtime <= index_mtime),
    }


def list_materials() -> list[dict[str, Any]]:
    """列出全部资料，按更新时间倒序。"""
    directory = _materials_dir()
    if not directory.exists():
        return []
    items = [_material_meta(path) for path in directory.glob("*.md")]
    return sorted(items, key=lambda item: item["updated_at"], reverse=True)


def read_material(name: str) -> dict[str, Any]:
    path = _path_for(name)
    if not path.exists():
        raise ProfileError(f"资料不存在：{safe_name(name)}")
    meta = _material_meta(path)
    meta["content"] = path.read_text(encoding="utf-8")
    return meta


def save_material(name: str, content: str, *, note: str = "") -> dict[str, Any]:
    """新建或覆盖一份资料；覆盖前自动留一份历史版本。"""
    path = _path_for(name)
    text = _require_content(content)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)

    if path.exists():
        _snapshot(path.stem, path.read_text(encoding="utf-8"), note)

    temp = path.with_suffix(".md.tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)
    return _material_meta(path)


def delete_material(name: str, *, note: str = "") -> None:
    """删除资料；删除前留一份版本，避免误删后无法恢复。"""
    path = _path_for(name)
    if not path.exists():
        raise ProfileError(f"资料不存在：{safe_name(name)}")
    _snapshot(path.stem, path.read_text(encoding="utf-8"), note or "delete")
    path.unlink()


def list_versions(name: str) -> list[dict[str, Any]]:
    directory = _version_dir(name)
    if not directory.exists():
        return []
    items = []
    for path in sorted(directory.glob("*.md"), reverse=True):
        note_path = path.with_suffix(".note")
        items.append(
            {
                "version": path.stem,
                "created_at": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "size": path.stat().st_size,
                "note": note_path.read_text(encoding="utf-8") if note_path.exists() else "",
            }
        )
    return items


def restore_version(name: str, version: str) -> dict[str, Any]:
    """把某个历史版本恢复成当前内容（恢复前的现状也会留档）。"""
    directory = _version_dir(name)
    candidate = directory / f"{safe_name(version)}.md"
    if not candidate.exists():
        raise ProfileError(f"版本不存在：{version}")
    return save_material(name, candidate.read_text(encoding="utf-8"), note=f"restore {version}")


def index_status() -> dict[str, Any]:
    """索引与资料的新鲜度，用于界面提示"要不要重建"。"""
    index_mtime = _index_mtime()
    newest = _newest_material_mtime()
    return {
        "materials": len(list(_materials_dir().glob("*.md"))) if _materials_dir().exists() else 0,
        "index_ready": index_mtime is not None,
        "index_updated_at": (
            datetime.fromtimestamp(index_mtime).strftime("%Y-%m-%d %H:%M:%S") if index_mtime else ""
        ),
        "stale": bool(index_mtime and newest and newest > index_mtime),
        "chunks": _index_chunk_count(),
    }


def _index_chunk_count() -> int:
    try:
        import faiss

        return int(faiss.read_index(str(INDEX_DIR / "index.faiss")).ntotal)
    except Exception:
        return 0


def _acquire_lock() -> None:
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    if LOCK_FILE.exists():
        age = time.time() - LOCK_FILE.stat().st_mtime
        if age < LOCK_TTL_SECONDS:
            raise ReindexBusy(f"已有重建任务在执行（{int(age)} 秒前开始）")
        LOCK_FILE.unlink(missing_ok=True)
    LOCK_FILE.write_text(str(os.getpid()), encoding="utf-8")


def rebuild_index(*, timeout: float = 300.0) -> dict[str, Any]:
    """重建向量索引，并清掉当前进程里的检索缓存。

    Returns:
        {"ok": bool, "chunks": int, "seconds": float, "output": str}

    Raises:
        ReindexBusy: 已有重建任务在跑。
    """
    _acquire_lock()
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, str(BUILD_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        ok = completed.returncode == 0
        if ok:
            # 关键一步：清掉进程内缓存，否则当前进程仍用旧索引对象。
            from knowledge import direct, retrieve

            direct.reset_cache()
            retrieve.reset_cache()
        return {
            "ok": ok,
            "chunks": _index_chunk_count(),
            "seconds": round(time.perf_counter() - started, 2),
            "output": output.strip()[-1200:],
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "chunks": _index_chunk_count(),
            "seconds": round(time.perf_counter() - started, 2),
            "output": f"重建超时（>{timeout:.0f}s）",
        }
    finally:
        LOCK_FILE.unlink(missing_ok=True)
