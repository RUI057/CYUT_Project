"""Gemini 雲端後端（選用）：需要 GEMINI_API_KEY 與 google-genai 套件。

語義修正的 prompt 與後端切換在 src/translate.py；這裡只負責呼叫 API。
延遲載入：沒裝 google-genai 或沒有金鑰時，不影響本機模型的使用。
"""
import os

from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-2.5-flash"
_client = None


def generate(system: str, contents: str) -> str:
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("未設定 GEMINI_API_KEY")
        from google import genai
        _client = genai.Client(api_key=key)
    from google.genai import types
    response = _client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system),
    )
    return response.text or ""
