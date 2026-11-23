"""候选人资料管理：服务层 + API 行为测试。

重点锁住三件事：
  1. 覆盖保存一定留历史版本（否则改错了没法回退）；
  2. 资料名不能穿越目录（name 会直接变成文件名）；
  3. 索引过期能被识别出来（这是"改了资料但产出还是旧事实"的根因）。
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app
from services import profile


class ProfileServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.materials = root / "docs"
        self.versions = root / "versions"
        self.index = root / "index"
        self.index.mkdir()
        for name in ("index.faiss", "index.pkl"):
            (self.index / name).write_text("x", encoding="utf-8")
        self.env = patch.dict(
            os.environ,
            {
                "CAREERPILOT_MATERIALS_DIR": str(self.materials),
                "CAREERPILOT_VERSIONS_DIR": str(self.versions),
            },
        )
        self.env.start()
        self.index_patch = patch.object(profile, "INDEX_DIR", self.index)
        self.index_patch.start()

    def tearDown(self) -> None:
        self.index_patch.stop()
        self.env.stop()
        self.tmp.cleanup()

    def test_create_then_read(self) -> None:
        meta = profile.save_material("项目经历", "# 项目\n- 做过 RAG", note="初版")
        self.assertEqual(meta["name"], "项目经历")
        self.assertEqual(meta["versions"], 0)
        detail = profile.read_material("项目经历")
        self.assertIn("做过 RAG", detail["content"])

    def test_save_keeps_previous_version(self) -> None:
        profile.save_material("简历", "版本一")
        profile.save_material("简历", "版本二", note="改数字")
        versions = profile.list_versions("简历")
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0]["note"], "改数字")
        detail = profile.read_material("简历")
        self.assertEqual(detail["content"], "版本二")

    def test_restore_version(self) -> None:
        profile.save_material("简历", "版本一")
        profile.save_material("简历", "版本二")
        version = profile.list_versions("简历")[0]["version"]
        profile.restore_version("简历", version)
        self.assertEqual(profile.read_material("简历")["content"], "版本一")
        # 恢复前的"版本二"也要留档
        self.assertEqual(len(profile.list_versions("简历")), 2)

    def test_delete_keeps_snapshot(self) -> None:
        profile.save_material("临时", "内容")
        profile.delete_material("临时")
        self.assertEqual(profile.list_materials(), [])
        self.assertEqual(len(profile.list_versions("临时")), 1)
        with self.assertRaises(profile.ProfileError):
            profile.read_material("临时")

    def test_name_traversal_is_rejected(self) -> None:
        self.assertEqual(profile.safe_name("../../etc/passwd"), "etc_passwd")
        with self.assertRaises(profile.ProfileError):
            profile.safe_name("   ")

    def test_empty_content_is_rejected(self) -> None:
        with self.assertRaises(profile.ProfileError):
            profile.save_material("空", "   ")

    def test_index_status_detects_stale(self) -> None:
        profile.save_material("简历", "内容")
        # 把索引时间调旧，模拟"资料比索引新"
        for name in ("index.faiss", "index.pkl"):
            os.utime(self.index / name, (1000, 1000))
        status = profile.index_status()
        self.assertTrue(status["stale"])
        self.assertEqual(status["materials"], 1)

    def test_index_status_fresh_after_touch(self) -> None:
        profile.save_material("简历", "内容")
        future = 2_000_000_000
        for name in ("index.faiss", "index.pkl"):
            os.utime(self.index / name, (future, future))
        self.assertFalse(profile.index_status()["stale"])


class ProfileApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.env = patch.dict(
            os.environ,
            {
                "CAREERPILOT_MATERIALS_DIR": str(root / "docs"),
                "CAREERPILOT_VERSIONS_DIR": str(root / "versions"),
            },
        )
        self.env.start()
        self.index_patch = patch.object(profile, "INDEX_DIR", root / "index")
        self.index_patch.start()
        (root / "index").mkdir()
        for name in ("index.faiss", "index.pkl"):
            (root / "index" / name).write_text("x", encoding="utf-8")
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.index_patch.stop()
        self.env.stop()
        self.tmp.cleanup()

    def test_material_crud_round_trip(self) -> None:
        created = self.client.post(
            "/profile/materials", json={"name": "简历", "content": "第一版", "note": "初版"}
        )
        self.assertEqual(created.status_code, 201)

        listed = self.client.get("/profile/materials")
        self.assertEqual([item["name"] for item in listed.json()], ["简历"])

        updated = self.client.put("/profile/materials/简历", json={"content": "第二版"})
        self.assertEqual(updated.status_code, 200)

        versions = self.client.get("/profile/materials/简历/versions")
        self.assertEqual(len(versions.json()), 1)
        version = versions.json()[0]["version"]
        restored = self.client.post(f"/profile/materials/简历/versions/{version}/restore")
        self.assertEqual(restored.status_code, 200)
        self.assertIn("第一版", self.client.get("/profile/materials/简历").json()["content"])

        deleted = self.client.delete("/profile/materials/简历")
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(self.client.get("/profile/materials/简历").status_code, 404)

    def test_blank_name_rejected(self) -> None:
        resp = self.client.post("/profile/materials", json={"name": "  ", "content": "x"})
        self.assertEqual(resp.status_code, 422)

    def test_index_status_endpoint(self) -> None:
        resp = self.client.get("/profile/index-status")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("stale", resp.json())

    def test_reindex_reports_busy(self) -> None:
        with patch.object(profile, "rebuild_index", side_effect=profile.ReindexBusy("占用中")):
            resp = self.client.post("/profile/reindex")
        self.assertEqual(resp.status_code, 409)


if __name__ == "__main__":
    unittest.main()
