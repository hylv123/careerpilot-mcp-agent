"""`agent/safety.py::redact_sensitive` 的单测。

为什么测这么细：
    脱敏是"宁可漏也不误伤"的活。误伤一条（比如把日期、版本号打成占位符），
    知识库内容就永久损坏；漏一条（比如身份证没被识别），它就是真的进了向量库。
    这两种错误都不会报错，只能靠用例守住。
"""

from __future__ import annotations

import unittest

from agent.safety import redact_sensitive


class RedactTests(unittest.TestCase):
    def test_phone_is_redacted(self) -> None:
        text, hits = redact_sensitive("联系电话 13812345678，随时可到岗")

        self.assertIn("[手机号]", text)
        self.assertNotIn("13812345678", text)
        self.assertEqual(hits.get("phone"), 1)

    def test_id_card_is_redacted_before_phone(self) -> None:
        """身份证 18 位要先整段吃掉，不能从里面截一段当手机号。"""
        text, hits = redact_sensitive("身份证 110101199003071234")

        self.assertIn("[身份证号]", text)
        self.assertNotIn("[手机号]", text)
        self.assertEqual(hits.get("id_card"), 1)

    def test_id_card_with_x_suffix(self) -> None:
        text, _ = redact_sensitive("证件号 11010119900307123X")

        self.assertIn("[身份证号]", text)
        self.assertNotIn("110101", text)

    def test_email_is_redacted(self) -> None:
        text, hits = redact_sensitive("邮箱 zhangsan@example.com")

        self.assertIn("[邮箱]", text)
        self.assertEqual(hits.get("email"), 1)

    def test_bank_card_is_redacted(self) -> None:
        text, hits = redact_sensitive("卡号 6222 0212 3456 7890")

        self.assertIn("[银行卡号]", text)
        self.assertEqual(hits.get("bank_card"), 1)

    def test_normal_text_is_untouched(self) -> None:
        """日期、版本号、普通数字不能被误伤。"""
        original = "2026-11-16 完成 v1.2.3，招聘 3 人，评分 20/20"
        text, hits = redact_sensitive(original)

        self.assertEqual(text, original)
        self.assertEqual(hits, {})

    def test_phone_inside_longer_number_is_not_matched(self) -> None:
        """长单号里恰好含 11 位手机号时，不能截出来替换。"""
        original = "订单号 913812345678012"
        text, hits = redact_sensitive(original)

        self.assertEqual(text, original)
        self.assertNotIn("phone", hits)

    def test_counters_accumulate_across_calls(self) -> None:
        counters: dict[str, int] = {}
        redact_sensitive("13812345678", counters)
        redact_sensitive("13900000000", counters)

        self.assertEqual(counters.get("phone"), 2)

    def test_multiple_types_in_one_text(self) -> None:
        text, hits = redact_sensitive(
            "张三 13812345678 zhangsan@example.com 110101199003071234"
        )

        self.assertIn("[手机号]", text)
        self.assertIn("[邮箱]", text)
        self.assertIn("[身份证号]", text)
        self.assertEqual(set(hits), {"phone", "email", "id_card"})


if __name__ == "__main__":
    unittest.main()
