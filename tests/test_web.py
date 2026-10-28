"""網頁 API：狀態、刪詞、設定保存與恢復預設、手語字典、首頁。不開攝影機、不載入語言模型。"""
import json
import os
import queue
import tempfile
import unittest
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)
import web  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class WebApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        web.SETTINGS_PATH = os.path.join(cls.tmp, "web_settings.json")
        cls.client = TestClient(web.app)          # 不用 with → 不跑 lifespan（不開執行緒、不預熱模型）

    def setUp(self):
        with web.HUB.lock:
            web.HUB.words = ["我", "要", "吃飯"]
            web.HUB.sentence = ""
            web.HUB.translating = False

    def tearDown(self):
        web.HUB.reset_settings()
        while True:                                # 清掉測試排進去的工作
            try:
                web.HUB.jobs.get_nowait()
                web.HUB.jobs.task_done()
            except queue.Empty:
                break
        with web.HUB.lock:
            web.HUB.translating = False

    def test_index_page(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(web.APP_NAME, r.text)
        for ph in ("<!--APP-->", "<!--VER-->", "<!--ORG-->"):
            self.assertNotIn(ph, r.text)

    def test_state(self):
        s = self.client.get("/state").json()
        for k in ("camera_on", "words", "sentence", "status", "conf_thresh", "auto_speak",
                  "model", "llm", "tts", "history", "last_word"):
            self.assertIn(k, s)
        self.assertEqual(s["words"], ["我", "要", "吃飯"])

    def test_remove_and_pop(self):
        self.assertTrue(self.client.post("/words/remove", json={"index": 1}).json()["ok"])
        self.assertEqual(web.HUB.words, ["我", "吃飯"])
        self.assertFalse(self.client.post("/words/remove", json={"index": 9}).json()["ok"])
        self.assertTrue(self.client.post("/words/pop").json()["ok"])
        self.assertEqual(web.HUB.words, ["我"])
        self.client.post("/words/pop")
        self.assertFalse(self.client.post("/words/pop").json()["ok"])

    def test_settings_clamp_persist_and_reset(self):
        default_conf = web.HUB.defaults["conf_thresh"]
        self.client.post("/settings", json={"conf_thresh": 5, "confirm_frames": 1, "auto_speak": False})
        s = self.client.get("/state").json()
        self.assertEqual(s["conf_thresh"], 1.0)           # 夾在 0.3～1.0
        self.assertEqual(s["confirm_frames"], 5)          # 夾在 5～30
        self.assertFalse(s["auto_speak"])
        with open(web.SETTINGS_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["confirm_frames"], 5)
        self.assertIn("model_sig", saved)

        web.HUB.apply_settings({"confirm_frames": 20})    # 模擬重開：從檔案載回
        web.HUB.load_settings()
        self.assertEqual(web.HUB.recognizer.confirm_frames, 5)

        self.client.post("/settings/reset")
        self.assertFalse(os.path.exists(web.SETTINGS_PATH))
        self.assertAlmostEqual(web.HUB.recognizer.conf_thresh, default_conf)
        self.assertTrue(web.HUB.auto_speak)

    def test_settings_from_other_model_keep_new_calibration(self):
        with open(web.SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump({"conf_thresh": 0.5, "confirm_frames": 9, "model_sig": "舊模型"}, f)
        web.HUB.load_settings()
        self.assertAlmostEqual(web.HUB.recognizer.conf_thresh, web.HUB.defaults["conf_thresh"])
        self.assertEqual(web.HUB.recognizer.confirm_frames, 9)

    def test_vocab(self):
        v = self.client.get("/vocab").json()
        self.assertGreater(v["n_total"], 0)
        self.assertLessEqual(v["n_known"], v["n_total"])
        words = [x for g in v["groups"] for x in g["words"]]
        self.assertEqual(len(words), v["n_total"])
        self.assertTrue(all({"w", "ok", "url", "hint"} <= set(x) for x in words))

    def test_speak_custom_text(self):
        r = self.client.post("/speak", json={"text": "  再念一次。 "}).json()
        self.assertTrue(r["ok"])
        kind, text = web.HUB.jobs.get_nowait()
        web.HUB.jobs.task_done()
        self.assertEqual((kind, text), ("speak", "再念一次。"))

    def test_speak_nothing(self):
        self.assertFalse(self.client.post("/speak").json()["ok"])   # 沒有句子也沒給文字


if __name__ == "__main__":
    unittest.main()
