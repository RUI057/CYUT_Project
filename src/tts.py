"""文字轉語音（gTTS，需要網路）。"""
import io
from typing import Optional

from gtts import gTTS


def synthesize(text: str, lang: str = "zh-TW") -> Optional[bytes]:
    """將文字合成為 mp3，回傳位元組（失敗回傳 None）。直接在記憶體合成，不寫暫存檔。"""
    if not text:
        return None
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang=lang, slow=False).write_to_fp(buf)
        return buf.getvalue()
    except Exception as e:
        print(f"[TTS 錯誤] {e}")
        return None
