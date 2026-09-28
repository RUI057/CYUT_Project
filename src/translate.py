"""語義修正：手語詞彙序列 → 自然繁體中文句子。

後端由環境變數 POLISH_BACKEND 決定（.env 或系統環境皆可）：
  local  （預設）本機模型，不需網路；失敗時若有 GEMINI_API_KEY 退回 Gemini
  gemini 雲端 Gemini；失敗時退回本機模型
  off    不修正，直接用「、」串接詞彙
所有後端都失敗時，一律退回串接，確保網頁不會因翻譯而中斷。
"""
import os
import re

from dotenv import load_dotenv

load_dotenv()

BACKEND = os.getenv("POLISH_BACKEND", "local").strip().lower()

SYSTEM = """你是台灣手語翻譯助手。使用者會給你手語辨識出的詞彙序列，
手語的語序和中文不同（常見：時間詞在前、動詞在後、疑問詞放句尾）。
請把詞彙序列改寫成一句自然通順的台灣繁體中文口語句子。
規則：
1. 只輸出一句句子，不要解釋、不要加引號。
2. 保留每個詞的意思，不要加入序列裡沒有的人名、數字、地點或事件。
3. 可以調整語序，並補上「的、了、要、在、嗎、很」等虛詞讓句子通順。
4. 疑問句用「？」結尾，其他用「。」結尾。
5. 一律使用繁體中文。"""

# 少樣本範例：示範手語語序 → 中文語序，以及「不要自己加資訊」
FEW_SHOT = [
    (["你", "名字", "什麼"], "你叫什麼名字？"),
    (["明天", "我", "學校", "去"], "我明天要去學校。"),
    (["我", "累"], "我好累。"),
    (["謝謝", "你", "幫忙"], "謝謝你的幫忙。"),
    (["爸爸", "生氣", "為什麼"], "爸爸為什麼生氣？"),
]

BACKEND_NAMES = {"local": "本機模型", "gemini": "Gemini", "off": "未修正"}
QUESTION_WORDS = ("什麼", "哪裡", "為什麼", "誰", "怎麼", "幾")

_cc = None


def _to_traditional(text):
    """轉成台灣繁體（小模型偶爾會冒出簡體字）。沒裝 opencc 就原樣回傳。"""
    global _cc
    if _cc is None:
        try:
            from opencc import OpenCC
            _cc = OpenCC("s2twp")
        except ImportError:
            print("[語義修正] 未安裝 opencc-python-reimplemented，略過簡轉繁")
            _cc = False
    return _cc.convert(text) if _cc else text


def _query(words):
    return "詞彙序列：" + "、".join(words)


def _messages(words):
    msgs = [{"role": "system", "content": SYSTEM}]
    for ws, sentence in FEW_SHOT:
        msgs += [{"role": "user", "content": _query(ws)},
                 {"role": "assistant", "content": sentence}]
    msgs.append({"role": "user", "content": _query(words)})
    return msgs


def _clean(text):
    """取第一行、去掉「翻譯：」之類前綴與引號。"""
    line = next((l.strip() for l in (text or "").splitlines() if l.strip()), "")
    line = re.sub(r"^(句子|翻譯|輸出|答案|中文)\s*[:：]\s*", "", line)
    return line.strip(" \"'「」“”『』")


def _finish(text):
    return _to_traditional(_clean(text))


def _issues(sentence, words):
    """需要請模型修正的地方：漏掉的詞、序列裡沒有卻自己加上的疑問詞。空 list 代表沒問題。"""
    joined = "".join(words)
    out = [f"漏掉了「{w}」" for w in words if w not in sentence]
    out += [f"序列裡沒有「{q}」，不要自己加入疑問詞"
            for q in QUESTION_WORDS if q in sentence and q not in joined]
    return out


def _plausible(sentence, words):
    """擋掉離題或亂編的輸出：太長、或原本的詞在句子裡幾乎找不到。"""
    if not sentence:
        return False
    joined = "".join(words)
    if len(sentence) > 3 * len(joined) + 12:
        return False
    # 每個詞有多少字出現在句子裡；平均低於一半視為離題（小模型改寫過頭或幻覺）
    cover = sum(sum(c in sentence for c in w) / len(w) for w in words) / len(words)
    return cover >= 0.5


def _local(words):
    from src import local_llm
    msgs  = _messages(words)
    first = _finish(local_llm.chat(msgs))
    issues = _issues(first, words)
    if not issues:
        return first
    # 自我修正一次：把問題回饋給模型重寫；真的變好才採用，避免越改越糟
    msgs += [{"role": "assistant", "content": first},
             {"role": "user", "content": "這句有問題：" + "；".join(issues) + "。請重新改寫成一句。"}]
    second = _finish(local_llm.chat(msgs))
    if len(_issues(second, words)) < len(issues) and _plausible(second, words):
        return second
    return first


def _gemini(words):
    from src import gemini_api
    return _finish(gemini_api.generate(SYSTEM, _query(words)))


_BACKENDS = {"local": _local, "gemini": _gemini}


def _order():
    has_key = bool(os.getenv("GEMINI_API_KEY"))
    if BACKEND == "off":
        return []
    if BACKEND == "gemini":
        return ["gemini", "local"]
    return ["local", "gemini"] if has_key else ["local"]


def polish(words):
    """詞彙序列 → (句子, 後端名稱)。所有後端失敗時回傳串接結果。"""
    words = [w for w in words if w]
    if not words:
        return "", BACKEND_NAMES["off"]
    for name in _order():
        try:
            sentence = _BACKENDS[name](words)
        except Exception as e:
            print(f"[語義修正] {BACKEND_NAMES[name]} 失敗：{e}")
            continue
        if _plausible(sentence, words):
            return sentence, BACKEND_NAMES[name]
        print(f"[語義修正] {BACKEND_NAMES[name]} 輸出不合理，改用下一個：{sentence!r}")
    return "、".join(words), BACKEND_NAMES["off"]


def polish_sentence(words) -> str:
    return polish(words)[0]


def warmup():
    """預先在背景載入本機模型（本機後端才需要）。"""
    if "local" in _order():
        from src import local_llm
        local_llm.warmup()


def backend_status() -> str:
    """給介面顯示的後端狀態，例如「本機模型（就緒）」。"""
    order = _order()
    if not order:
        return BACKEND_NAMES["off"]
    if order[0] == "local":
        from src import local_llm
        return f"本機模型 {local_llm.MODEL_ID.split('/')[-1]}（{local_llm.status()}）"
    return "Gemini（雲端）"
