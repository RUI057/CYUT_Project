"""詞彙：示範影片對應；語義修正：單一詞的規則（不載入語言模型）。"""
import os
import tempfile
import unittest

from src import translate
from src.vocab import get_all_labels, get_demo_links


class VocabTests(unittest.TestCase):
    def test_every_word_has_demo_video(self):
        links = get_demo_links()
        missing = [w for w in get_all_labels() if w not in links]
        self.assertEqual(missing, [], f"這些詞在手語演示網址.txt 找不到影片：{missing}")

    def test_label_formats(self):
        text = ("你我他\nhttps://a\n\n"
                "一 到 十\nhttps://b\n\n"
                "上課（近義：課南）\nhttps://c\n\n"
                "要 (北部)\nhttps://d\n\n"
                "請 謝謝 對不起\nhttps://e\n\n"
                "四\nhttps://f\n")
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(text)
        try:
            d = get_demo_links(f.name)
        finally:
            os.remove(f.name)
        self.assertEqual(d["他"], "https://a")
        self.assertEqual(d["七"], "https://b")
        self.assertEqual(d["四"], "https://b")          # 重複出現時取第一段
        self.assertEqual(d["上課"], "https://c")
        self.assertEqual(d["要"], "https://d")
        self.assertEqual(d["對不起"], "https://e")

    def test_missing_file(self):
        self.assertEqual(get_demo_links("/nonexistent/demo.txt"), {})


class SingleWordRuleTests(unittest.TestCase):
    def test_rules(self):
        if translate.BACKEND == "off":
            self.skipTest("POLISH_BACKEND=off")
        self.assertEqual(translate.polish(["誰"])[0], "誰？")
        self.assertEqual(translate.polish(["你好"])[0], "你好！")
        self.assertEqual(translate.polish(["累"])[0], "累。")
        self.assertEqual(translate.polish([])[0], "")


if __name__ == "__main__":
    unittest.main()
