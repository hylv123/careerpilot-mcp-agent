"""经历原子：抽取、确认、防篡改。

核心断言只有一句：**原子里的 span 必须能在来源文件里逐字找到**。
确认时再核一次，是为了防止"抽出来之后有人改了文档"导致原子变成过期引用。
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app
from services import atoms


class AtomServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.docs = root / "docs"
        self.docs.mkdir()
        self.source = self.docs / "项目经历.md"
        self.source.write_text(
            "# 项目经历\n\n## CareerPilot\n\n- 用 LangGraph 编排 9 节点工作流，三路并发生成\n"
            "- 用 Redis Stream 做异步任务队列，PEL 回收滞留任务\n",
            encoding="utf-8",
        )
        self.env = patch.dict(
            os.environ, {"CAREERPILOT_ATOMS_PATH": str(root / "atoms.json")}
        )
        self.env.start()
        self.docs_patch = patch("knowledge.corpus.indexed_files", return_value=[self.source])
        self.docs_patch.start()
        self.root_patch = patch.object(atoms, "PROJECT_ROOT", root)
        self.root_patch.start()

    def tearDown(self) -> None:
        self.root_patch.stop()
        self.docs_patch.stop()
        self.env.stop()
        self.tmp.cleanup()

    def test_extract_splits_bullets_verbatim(self) -> None:
        result = atoms.extract_candidates()
        self.assertEqual(result["added"], 2)
        stored = atoms.load_atoms()
        self.assertEqual(stored[0]["span"], "用 LangGraph 编排 9 节点工作流，三路并发生成")
        self.assertEqual(stored[0]["project"], "CareerPilot")
        self.assertFalse(stored[0]["approved"], "抽取出来默认不能直接用")

    def test_span_must_exist_in_source(self) -> None:
        atoms.extract_candidates()
        atom = atoms.load_atoms()[0]
        self.assertTrue(atoms.verify_span(atom["source"], atom["span"]))
        self.assertFalse(atoms.verify_span(atom["source"], "这句原文里没有"))

    def test_approve_and_stats(self) -> None:
        atoms.extract_candidates()
        atom = atoms.load_atoms()[0]
        approved = atoms.set_approved(atom["id"], True)
        self.assertTrue(approved["approved"])
        self.assertEqual(
            atoms.stats(),
            {"total": 2, "approved": 1, "pending": 1, "experience": 2, "knowledge": 0},
        )

    def test_approve_rejects_tampered_span(self) -> None:
        atoms.extract_candidates()
        stored = atoms.load_atoms()
        stored[0]["span"] = "这段原文里根本没有"
        atoms.save_atoms(stored)
        with self.assertRaises(atoms.AtomError):
            atoms.set_approved(stored[0]["id"], True)

    def test_extract_is_idempotent(self) -> None:
        atoms.extract_candidates()
        second = atoms.extract_candidates()
        self.assertEqual(second["added"], 0)
        self.assertEqual(second["skipped"], 2)

    def test_delete_atom(self) -> None:
        atoms.extract_candidates()
        atom = atoms.load_atoms()[0]
        atoms.delete_atom(atom["id"])
        self.assertEqual(len(atoms.load_atoms()), 1)
        with self.assertRaises(atoms.AtomError):
            atoms.delete_atom(atom["id"])


class AtomApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.docs = root / "docs"
        self.docs.mkdir()
        self.source = self.docs / "简历.md"
        self.source.write_text("# 简历\n\n- 独立开发求职辅助智能体，覆盖 JD 分析与材料生成\n", encoding="utf-8")
        self.env = patch.dict(os.environ, {"CAREERPILOT_ATOMS_PATH": str(root / "atoms.json")})
        self.env.start()
        self.docs_patch = patch("knowledge.corpus.indexed_files", return_value=[self.source])
        self.docs_patch.start()
        self.root_patch = patch.object(atoms, "PROJECT_ROOT", root)
        self.root_patch.start()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.root_patch.stop()
        self.docs_patch.stop()
        self.env.stop()
        self.tmp.cleanup()

    def test_extract_list_approve_delete(self) -> None:
        extracted = self.client.post("/profile/atoms/extract").json()
        self.assertEqual(extracted["added"], 1)

        listed = self.client.get("/profile/atoms").json()
        self.assertEqual(listed["stats"]["pending"], 1)
        atom_id = listed["atoms"][0]["id"]

        approved = self.client.patch(f"/profile/atoms/{atom_id}", json={"approved": True})
        self.assertEqual(approved.status_code, 200)
        self.assertTrue(approved.json()["approved"])

        filtered = self.client.get("/profile/atoms", params={"approved": "true"}).json()
        self.assertEqual(len(filtered["atoms"]), 1)

        self.assertEqual(self.client.delete(f"/profile/atoms/{atom_id}").status_code, 204)
        self.assertEqual(self.client.delete(f"/profile/atoms/{atom_id}").status_code, 404)


if __name__ == "__main__":
    unittest.main()
