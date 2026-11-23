"""经历原子：把简历/项目文档切成"可逐字核对"的事实单元。

为什么需要它：
    现在的真实性校验只到"文件名级引用"——模型说这句话来自 project.md，
    只要文件名对得上就算过。文件里可能有一百句，它到底抄了哪句没人知道。
    原子把粒度压到"一句原文"：span 必须是从文档里**逐字**摘出来的子串，
    于是"引用"第一次变成可机器验证的事。

为什么抽取先做确定性切分而不是让模型总结：
    模型总结会顺手改写——"参与"变"负责"、补一个数字，正好把要防的东西
    带进来。这里只按标题和列表项切分，span 原样保留；改写留到生成阶段，
    由 span 校验和主张强度校验兜着。

人工确认是硬闸门：
    抽出来的原子默认 approved=false，只有人点过才会进入可引用池。
    宁可慢一步，也不让"机器觉得像事实"的句子直接进简历。
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ATOMS_PATH = PROJECT_ROOT / "data" / "evidence_atoms.json"

_HEADING = re.compile(r"^#{1,4}\s+(.*\S)\s*$")
_BULLET = re.compile(r"^\s*[-*]\s+(.+?)\s*$")
# 段落里的事实句也进候选：只认带数字或英文技术词的句子，避免把纯议论抽成原子
_SENTENCE = re.compile(r"[^。！？；\n]{12,120}[。！？；]?")
_FACTUAL = re.compile(r"[0-9]|[A-Za-z]{3,}")
_WS = re.compile(r"\s+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


class AtomError(ValueError):
    """原子层的可预期错误。"""


def _norm(text: str) -> str:
    return _WS.sub("", str(text or "")).casefold()


def _atom_id(source: str, span: str) -> str:
    digest = hashlib.sha1(f"{source}|{_norm(span)}".encode("utf-8")).hexdigest()
    return f"atom-{digest[:12]}"


def _atoms_path() -> Path:
    override = __import__("os").getenv("CAREERPILOT_ATOMS_PATH", "").strip()
    return Path(override).resolve() if override else ATOMS_PATH


def load_atoms() -> list[dict[str, Any]]:
    path = _atoms_path()
    if not path.exists():
        return []
    try:
        return list(json.loads(path.read_text(encoding="utf-8")).get("atoms", []))
    except (json.JSONDecodeError, OSError) as exc:
        raise AtomError(f"原子库读取失败：{exc}") from exc


def save_atoms(atoms: list[dict[str, Any]]) -> None:
    path = _atoms_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "updated_at": datetime.now().isoformat(timespec="seconds"), "atoms": atoms}
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _source_text(source: str) -> str:
    path = PROJECT_ROOT / source
    if not path.exists():
        raise AtomError(f"来源文件不存在：{source}")
    return path.read_text(encoding="utf-8")


def verify_span(source: str, span: str) -> bool:
    """span 必须是来源文件里的逐字子串（忽略空白与大小写差异）。"""
    try:
        return _norm(span) in _norm(_source_text(source))
    except AtomError:
        return False


_EXPERIENCE_HINTS = ("private/", "项目", "简历")


def _atom_kind(source: str, project: str) -> str:
    """区分"我的经历"和"通用知识"。

    FAISS 语料里混着两类东西：private/ 下的简历项目，以及课程/面试笔记。
    后者是知识不是经历，可以拿来准备面试，但不能写进简历说"我做过"。
    """
    haystack = f"{source} {project}".casefold()
    return "experience" if any(hint in haystack for hint in _EXPERIENCE_HINTS) else "knowledge"


def _candidate_spans(line: str) -> list[str]:
    """一行里可能的原子：整条列表项，以及句中带数字/技术词的事实句。"""
    bullet = _BULLET.match(line)
    if bullet:
        text = bullet.group(1).strip()
        return [text] if len(text) >= 8 else []
    if not _FACTUAL.search(line):
        return []
    return [match.group(0).strip(" 。；") for match in _SENTENCE.finditer(line)]


def extract_candidates() -> dict[str, Any]:
    """从**全部参与索引的文档**里切出候选原子（不改写、不补全）。

    范围必须与 FAISS 索引一致（knowledge/corpus.py）：只扫私有资料时，
    公开笔记里的事实进不了原子库，"只引用原子"就会凭空砍掉可用经历。
    """
    from knowledge.corpus import indexed_files

    atoms = load_atoms()
    known = {atom["id"] for atom in atoms}
    added: list[dict[str, Any]] = []
    skipped = 0

    for path in indexed_files():
        source = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        project = path.stem
        for line in path.read_text(encoding="utf-8").splitlines():
            heading = _HEADING.match(line)
            if heading:
                project = heading.group(1).strip()
                continue
            for span in _candidate_spans(line):
                atom_id = _atom_id(source, span)
                if atom_id in known:
                    skipped += 1
                    continue
                known.add(atom_id)
                added.append(
                    {
                        "id": atom_id,
                        "project": project,
                        "span": span,
                        "source": source,
                        "kind": _atom_kind(source, project),
                        "metrics": _NUMBER.findall(span),
                        "approved": False,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                    }
                )

    if added:
        save_atoms([*atoms, *added])
    return {"added": len(added), "skipped": skipped, "total": len(atoms) + len(added)}


def list_atoms(*, approved: bool | None = None) -> list[dict[str, Any]]:
    atoms = load_atoms()
    if approved is not None:
        atoms = [atom for atom in atoms if bool(atom.get("approved")) is approved]
    return atoms


def set_approved(atom_id: str, approved: bool) -> dict[str, Any]:
    atoms = load_atoms()
    for atom in atoms:
        if atom["id"] == atom_id:
            if approved and not verify_span(atom["source"], atom["span"]):
                raise AtomError("证据片段与来源文件不一致，不能确认")
            atom["approved"] = bool(approved)
            atom["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
            save_atoms(atoms)
            return atom
    raise AtomError(f"原子不存在：{atom_id}")


def approve_bulk(*, kind: str | None = None, source: str | None = None) -> int:
    """批量确认；只确认片段仍与原文一致的原子，返回确认条数。"""
    atoms = load_atoms()
    changed = 0
    for atom in atoms:
        if atom.get("approved"):
            continue
        if kind and atom.get("kind", "experience") != kind:
            continue
        if source and not str(atom.get("source", "")).endswith(source):
            continue
        if not verify_span(atom["source"], atom["span"]):
            continue
        atom["approved"] = True
        atom["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
        changed += 1
    if changed:
        save_atoms(atoms)
    return changed


def delete_atom(atom_id: str) -> None:
    atoms = load_atoms()
    remaining = [atom for atom in atoms if atom["id"] != atom_id]
    if len(remaining) == len(atoms):
        raise AtomError(f"原子不存在：{atom_id}")
    save_atoms(remaining)


def approved_experience_atoms() -> list[dict[str, Any]]:
    """已确认、且属于"我的经历"的原子——生成阶段唯一允许当事实用的池子。"""
    return [
        atom
        for atom in load_atoms()
        if atom.get("approved") and atom.get("kind", "experience") == "experience"
    ]


def stats() -> dict[str, int]:
    atoms = load_atoms()
    approved = sum(1 for atom in atoms if atom.get("approved"))
    experience = sum(1 for atom in atoms if atom.get("kind", "experience") == "experience")
    return {
        "total": len(atoms),
        "approved": approved,
        "pending": len(atoms) - approved,
        "experience": experience,
        "knowledge": len(atoms) - experience,
    }
