import json
import os

VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "vocabulary.json")
HINTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "hints.json")

_hints_cache = None

def load_vocab() -> dict:
    """讀取完整 vocabulary.json"""
    with open(VOCAB_PATH, encoding="utf-8") as f:
        return json.load(f)

def get_all_labels() -> list[str]:
    """取得所有詞彙的扁平清單，給模型訓練用"""
    vocab = load_vocab()
    return [label for group in vocab["groups"] for label in group["labels"]]

def get_labels_by_priority(priority: str) -> list[str]:
    """依優先度篩選，priority = '高' / '中' / '低'"""
    vocab = load_vocab()
    return [
        label
        for group in vocab["groups"]
        if group["priority"] == priority
        for label in group["labels"]
    ]

def get_group_names() -> list[str]:
    """取得所有分組名稱"""
    return [g["name"] for g in load_vocab()["groups"]]

def get_hint(label: str) -> str:
    """取得詞彙的比法提示（data/hints.json）；沒有該詞或檔案不存在時回傳空字串"""
    global _hints_cache
    if _hints_cache is None:
        try:
            with open(HINTS_PATH, encoding="utf-8") as f:
                _hints_cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _hints_cache = {}
    # 從其他資料貼上的提示常以冒號開頭（「名字：…」只複製到冒號後），畫面會變成「提示：：…」
    return _hints_cache.get(label, "").strip().lstrip("：:").strip()

DEMO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "手語演示網址.txt")
_NUMS = "一二三四五六七八九十"


def get_demo_links(path: str = DEMO_PATH) -> dict:
    """讀示範影片清單（data/手語演示網址.txt，每段「標籤\\n網址」）→ {詞: 網址}。

    標籤有幾種寫法，都要能對到詞彙表裡的詞：
      單一詞「學校」、空白分隔多個詞「請 謝謝 對不起」、括號註記「上課（近義：課南）」「要 (北部)」、
      數字範圍「一 到 十」、連寫的單字詞「你我他」。
    同一個詞出現在多段時取第一段；檔案不存在回傳空 dict。
    """
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return {}
    vocab = set(get_all_labels())
    links = {}
    for block in text.strip().split("\n\n"):
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if len(lines) < 2 or not lines[1].startswith("http"):
            continue
        label, url = lines[0], lines[1]
        base = label
        for a, b in (("（", "）"), ("(", ")")):
            while a in base and b in base.split(a, 1)[1]:
                head, rest = base.split(a, 1)
                base = head + rest.split(b, 1)[1]
        base = base.strip()
        words = []
        rng = [t for t in base.split() if t != "到"]
        if "到" in base.split() and len(rng) == 2 and all(t in _NUMS for t in rng):
            lo, hi = _NUMS.index(rng[0]), _NUMS.index(rng[1])
            words = list(_NUMS[lo:hi + 1])
        else:
            for tok in base.replace("、", " ").split():
                if tok in vocab:
                    words.append(tok)
                elif len(tok) > 1 and all(ch in vocab for ch in tok):   # 「你我他」
                    words.extend(tok)
        for w in words:
            links.setdefault(w, url)
    return links
