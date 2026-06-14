import os
from gtts import gTTS
import streamlit as st


def synthesize(text: str, lang: str = "zh-TW") -> bytes | None:
    """將文字合成為 mp3，回傳位元組（失敗回傳 None）。不直接渲染。"""
    if not text:
        return None
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        os.makedirs("history", exist_ok=True)
        audio_path = "history/latest_tts.mp3"
        tts.save(audio_path)
        with open(audio_path, "rb") as f:
            return f.read()
    except Exception as e:
        print(f"[TTS 錯誤] {e}")
        return None


def speak(text: str, lang: str = "zh-TW", container=None) -> None:
    """
    將文字轉成語音並在 Streamlit 頁面播放。
    使用 gTTS（Google Text-to-Speech），支援繁體中文。
    container：可傳入 st.empty() 佔位元素，避免在迴圈中重複堆疊播放器。
    """
    if not text:
        return

    audio_bytes = synthesize(text, lang)
    if audio_bytes is None:
        st.warning("語音合成失敗，請檢查網路連線")
        return

    target = container if container is not None else st
    target.audio(audio_bytes, format="audio/mp3", autoplay=True)
