"""語義修正：手語詞彙序列 → 自然繁體中文句子。

後端由環境變數 POLISH_BACKEND 決定（.env 或系統環境皆可）：
  local  （預設）本機模型，不需網路；失敗時若有 GEMINI_API_KEY 退回 Gemini
  gemini 雲端 Gemini；失敗時退回本機模型
  off    不修正，直接用「、」串接詞彙
所有後端都失敗時，一律退回串接，確保網頁不會因翻譯而中斷。

準確度（2026-10-25 評測，見 _local 說明）：
  - 單一個詞不問模型，直接用規則輸出（真實使用約 6 成是單詞，模型在這裡只會亂補）
  - 本機模型用束搜尋產生多個候選，再依「忠實度」挑一個，而不是只取貪婪解碼的第一名
  POLISH_BEAMS 控制候選數（預設 6；1 = 關閉，回到貪婪解碼）。
"""
import os
import re

from dotenv import load_dotenv

load_dotenv()

BACKEND = os.getenv("POLISH_BACKEND", "local").strip().lower()
BEAMS   = max(1, int(os.getenv("POLISH_BEAMS", "6")))

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
    # 問路情境：方向（往 + 方向 + 走）、位置（X 在 + 方位）。
    # 2026-10-01 實測：沒有這兩例時「左 走路」→「我左邊走路」、「右 走路 廁所」會亂編「請注意安全」；
    # 另試過「請 等 → 請等一下」，但模型會把「等一下」照抄進不相干的句子，所以沒有採用。
    (["右", "走路"], "往右走。"),
    (["廁所", "前面"], "廁所在前面。"),
]

BACKEND_NAMES = {"local": "本機模型", "gemini": "Gemini", "off": "未修正", "rule": "規則"}
QUESTION_WORDS = ("什麼", "哪裡", "為什麼", "誰", "怎麼", "幾")

# 同一個手勢對應多個中文詞：模型只會收到前者，但句子寫成後者也算保留了這個詞。
# （「今天」「現在」比法相同，2026-10-25 合併成同一個詞，見 data/vocabulary.json）
# 提示詞裡加「今天也可寫成現在」實測無效：本機 1.5B 模型一次都沒用到「現在」，
# 還讓「右 走路」從「往右走」變成「我往右走」，所以只在檢查邏輯認同義詞，不動提示詞。
# 「幫忙」是不及物動詞，通順的說法是「幫我忙」或「幫我」，不是「幫忙我」。
# 「什麼」放在主詞後面沒有動詞時，通順的說法是「做什麼」「幹什麼」（「你晚上什麼？」不通）。
# 長的放前面：_rest 依序刪除，才不會把「做什麼」只刪掉「什麼」而留下「做」被當成多加的字。
SYNONYMS = {"今天": ("今天", "現在"),
            "幫忙": ("幫忙", "幫我忙", "幫你忙", "幫他忙", "幫"),
            "什麼": ("做什麼", "幹什麼", "什麼")}


def _alts(word):
    return SYNONYMS.get(word, (word,))

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


_NON_CJK = re.compile(r"[A-Za-z\u3040-\u30ff]+")      # 小模型偶爾會冒出 want 之類的英文

# 小模型自己多加、會改變語意的動詞／助動詞（「我 讀書」→「我要讀書」、「我 不會 手語」→「我不會說手語」）。
# 輸入裡沒有、句子裡卻有，就在排序時排後面（嚴格分層，不看模型信心）。
_ADD_WATCH = ("要", "想", "說", "會", "去", "來", "吃", "喝", "買", "給", "拿", "坐", "找", "問", "讀", "看", "聽")
# 自由補的虛詞字與標點：不算「多加」
_FUNC = set("的了在很好太真是有和跟與也都就吧呢啊著過把被對叫")
_PUNCT = set("。？！，、,.?! ")
_GREETINGS = {"你好", "再見", "早安"}


def _rest(sentence, words):
    """扣掉輸入詞（含同義詞）之後，句子還剩下什麼——剩下的就是模型自己加的。"""
    for w in sorted(words, key=len, reverse=True):
        for a in _alts(w):
            sentence = sentence.replace(a, "")
    return sentence


def _added_verbs(sentence, words):
    rest = _rest(sentence, words)
    return [c for c in _ADD_WATCH if c in rest]


def _extra_chars(sentence, words):
    """不屬於輸入詞、也不是常見虛詞／標點的字數：越少越忠實。"""
    return sum(1 for c in _rest(sentence, words) if c not in _FUNC and c not in _PUNCT)


def _inversions(sentence, words):
    """句子裡各詞出現的順序和輸入順序相比，有幾對顛倒。手語的詞序本身帶有關係
    （「明天 考試 我 不會」≠「我明天不會考試」），能保持原序就保持。"""
    pos = []
    for w in words:
        idx = [sentence.find(a) for a in _alts(w) if a in sentence]
        if idx:
            pos.append(min(idx))
    return sum(1 for i in range(len(pos)) for j in range(i + 1, len(pos)) if pos[i] > pos[j])


def _single(word):
    """單一個詞：疑問詞加「？」，其餘原詞加句號（招呼語用「！」）。不問模型，不會亂補。"""
    if any(q in word for q in QUESTION_WORDS):
        return word + "？"
    return word + ("！" if word in _GREETINGS else "。")


def _issues(sentence, words):
    """需要請模型修正的地方：漏掉的詞、序列裡沒有卻自己加上的疑問詞。空 list 代表沒問題。"""
    joined = "".join(words)
    out = [f"漏掉了「{w}」" for w in words if not any(a in sentence for a in _alts(w))]
    out += [f"序列裡沒有「{q}」，不要自己加入疑問詞"
            for q in QUESTION_WORDS if q in sentence and q not in joined]
    out += [f"不要使用英文或日文「{x}」，只能用中文" for x in _NON_CJK.findall(sentence)]
    return out


def _plausible(sentence, words):
    """擋掉離題或亂編的輸出：太長、或原本的詞在句子裡幾乎找不到。"""
    if not sentence:
        return False
    joined = "".join(words)
    if len(sentence) > 3 * len(joined) + 12:
        return False
    # 每個詞有多少字出現在句子裡；平均低於一半視為離題（小模型改寫過頭或幻覺）
    cover = sum(max(sum(c in sentence for c in a) / len(a) for a in _alts(w))
                for w in words) / len(words)
    return cover >= 0.5


def _rank_key(sentence, score, words):
    """候選排序（越小越好）。分層：先看硬問題（漏詞、多疑問詞、英文），
    再看自己多加的動詞，同層才用「模型信心 − 多加字數」。

    2026-10-25 在 71 句（調參集＋保留集，詞序、時間、疑問、稱謂等）的評測：
      貪婪解碼 78%／69% → 束搜尋＋這個排序 94%／89%（調參集／保留集）；
      另外盲寫的 25 句驗證集 56% → 84%。
    消融：多加字懲罰從 0.5 降到 0.1 保留集掉 6 點、「幫忙」同義詞掉 3 點；「嗎」猜測、
    「幫忙我」文法兩種懲罰沒有可量測的貢獻，所以沒放進來。詞序懲罰在精確比對的分數上看不出差別，
    但它擋掉了「明天 考試 我 不會」→「我明天不會考試」這種改變語意的重排，所以保留。
    """
    return ((len(_issues(sentence, words)), len(_added_verbs(sentence, words))),
            -score + 0.5 * _extra_chars(sentence, words) + 0.05 * _inversions(sentence, words))


def _local(words):
    from src import local_llm
    msgs = _messages(words)
    if BEAMS > 1:
        cands = [(_finish(t), sc) for t, sc in local_llm.chat_candidates(msgs, BEAMS)]
        cands = [(t, sc) for t, sc in cands if t and _plausible(t, words)]
        first = min(cands, key=lambda c: _rank_key(c[0], c[1], words))[0] if cands else None
    else:
        first = None
    if first is None:                       # 候選全不合理（或關閉束搜尋）→ 貪婪解碼
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
    if len(words) == 1 and BACKEND != "off":
        return _single(words[0]), BACKEND_NAMES["rule"]
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
