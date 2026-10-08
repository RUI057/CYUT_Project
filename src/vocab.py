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