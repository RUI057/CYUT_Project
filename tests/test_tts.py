"""朗讀：格式判斷、線上失敗時改用本機語音（不實際連網）。"""
import os
import unittest
from unittest import mock

from src import tts


class TtsTests(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get("TTS_BACKEND")
        tts._online_down_until = 0.0

    def tearDown(self):
        if self._env is None:
            os.environ.pop("TTS_BACKEND", None)
        else:
            os.environ["TTS_BACKEND"] = self._env
        tts._online_down_until = 0.0

    def test_mime(self):
        self.assertEqual(tts.mime_of(b"RIFF....WAVE"), "audio/wav")
        self.assertEqual(tts.mime_of(b"\xff\xf3\x84"), "audio/mpeg")

    def test_empty_text(self):
        self.assertIsNone(tts.synthesize(""))

    def test_auto_falls_back_and_remembers(self):
        os.environ["TTS_BACKEND"] = "auto"
        with mock.patch.object(tts, "_online", return_value=None) as on, \
             mock.patch.object(tts, "_offline", return_value=b"RIFFfake") as off:
            self.assertEqual(tts.synthesize("你好。"), b"RIFFfake")
            self.assertEqual(tts.synthesize("再見。"), b"RIFFfake")
        self.assertEqual(on.call_count, 1)        # 第二句不再等線上逾時
        self.assertEqual(off.call_count, 2)

    def test_online_mode_never_uses_offline(self):
        os.environ["TTS_BACKEND"] = "online"
        with mock.patch.object(tts, "_online", return_value=None), \
             mock.patch.object(tts, "_offline", return_value=b"RIFFfake") as off:
            self.assertIsNone(tts.synthesize("你好。"))
        off.assert_not_called()

    def test_offline_mode_skips_online(self):
        os.environ["TTS_BACKEND"] = "offline"
        with mock.patch.object(tts, "_online") as on, \
             mock.patch.object(tts, "_offline", return_value=b"RIFFfake"):
            self.assertEqual(tts.synthesize("你好。"), b"RIFFfake")
        on.assert_not_called()

    @unittest.skipUnless(tts.offline_available(), "這台電腦沒有本機中文語音（只支援 macOS）")
    def test_real_offline_voice(self):
        data = tts._offline("測試。")
        self.assertTrue(data and data[:4] == b"RIFF" and len(data) > 2000)


if __name__ == "__main__":
    unittest.main()
