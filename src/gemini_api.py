import os
import base64
import numpy as np
from PIL import Image
import io
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PROMPT = """你是台灣手語辨識專家。
圖片中的人正在打手語，請辨識這個手勢。
規則：
1. 只辨識【台灣手語（TSL）】詞彙或指拼字母
2. 只回傳一個詞彙或一個字母，例如：你好、謝謝、吃飯、A、B
3. 不要加任何解釋、標點或換行
4. 完全無法判斷時，只回傳「？」"""

def _array_to_bytes(frame_rgb: np.ndarray) -> bytes:
    img = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

def recognize_sign(camera_input) -> str:
    """給 Streamlit camera_input 用"""
    try:
        img_bytes = camera_input.getvalue()
        return _call_gemini(img_bytes)
    except Exception as e:
        print(f"[Gemini 錯誤] {e}")
        return "？"

def recognize_sign_from_array(frame_rgb: np.ndarray) -> str:
    """給即時 OpenCV frame 用"""
    try:
        img_bytes = _array_to_bytes(frame_rgb)
        return _call_gemini(img_bytes)
    except Exception as e:
        print(f"[Gemini 錯誤] {e}")
        return "？"

def _call_gemini(img_bytes: bytes) -> str:
    response = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            PROMPT,
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
        ]
    )
    word = response.text.strip().replace("\n", "").replace(" ", "")
    return word if word else "？"
