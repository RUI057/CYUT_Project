import os
from gtts import gTTS
import streamlit as st

def speak(text: str, lang: str = "zh-TW") -> None:
    """
    將文字轉成語音並在 Streamlit 頁面播放。
    使用 gTTS（Google Text-to-Speech），支援繁體中文。
    """
    if not text:
        return

    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_path = "history/latest_tts.mp3"
        tts.save(audio_path)

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        # 在 Streamlit 頁面顯示音訊播放器
        st.audio(audio_bytes, format="audio/mp3", autoplay=True)

    except Exception as e:
        print(f"[TTS 錯誤] {e}")
        st.warning("語音合成失敗，請檢查網路連線")
