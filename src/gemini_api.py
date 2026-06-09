import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM = """你是一個專業的台灣手語翻譯助手。
手語的語法和中文不同（通常是「主題-評論」結構），
使用者會給你一串手語辨識出的詞彙序列，
你的任務是將它轉換成自然流暢的繁體中文口語句子。
規則：
- 只輸出翻譯後的句子，不要加任何解釋
- 保持原意，不要增加或刪減資訊
- 使用日常口語、繁體中文
- 若詞彙不足以成句，盡力合理推斷補全"""

def polish_sentence(word_list: list) -> str:
    if not word_list:
        return ""
    words_str = "、".join(word_list)
    try:
        response = _client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"詞彙序列：{words_str}",
            config=types.GenerateContentConfig(system_instruction=SYSTEM)
        )
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini Pro 錯誤] {e}")
        return "、".join(word_list)