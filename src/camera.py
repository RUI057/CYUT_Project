"""攝影機端手語辨識：載入 LSTM 模型、即時推理、穩定確認。

架構與 train_model.py 一致（雙向 LSTM + mean/max 池化），
特徵抽取與尺度正規化共用 src/features，確保訓練/推理一致。
特徵只有雙手 126 維（臉部特徵已於 2026-10 整個拿掉）。
舊模型（2026-10 以前訓練）輸入是 162 維：後 36 維是補零的臉部，載入時照舊補零，
輸出與當年完全相同，不必馬上重訓；重訓後新模型就是 126 維。
"""
import os
import sys
import pickle
from collections import deque

import numpy as np
import torch
import mediapipe as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import (extract_two_hands, to_model_input, FEAT_DIM,
                          clean_hand_tracks, hand_motion, MOTION_THRESH,
                          MP_HANDS_KWARGS)
from src.model import GestureLSTM

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils


# ── 推理裝置（LSTM 定義集中於 src/model.py）──────────────
if torch.backends.mps.is_available():
    _device = torch.device("mps")
elif torch.cuda.is_available():
    _device = torch.device("cuda")
else:
    _device = torch.device("cpu")

# ── 載入模型 ──────────────────────────────────
# 用本檔位置找模型，不依賴「從哪個資料夾啟動」（否則從別處啟動會誤報找不到模型）
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "model.pkl")
_model = None
_classes = []
_seq_len = 30
_clean_tracks = False  # 由 model.pkl 的 clean_tracks 決定；舊模型訓練時沒濾波就不濾
_in_dim = FEAT_DIM # 由 model.pkl 的 feat_dim 決定；大於 126 的舊模型，多出來的維度補 0
_motion_thr = {}   # 由 model.pkl 的 motion_thresh 決定：每個詞的動作門檻（靜態手勢較低）
DEFAULT_CONF = 0.8 # 模型沒有校準過的信心門檻時（舊模型、--quick 訓練）才用
TOP_K = 3          # 回報給介面的候選詞數量
_conf_thr = None   # 由 model.pkl 的 conf_thresh 決定：訓練時用交叉驗證校準的信心門檻

try:
    with open(MODEL_PATH, "rb") as f:
        _payload = pickle.load(f)
    _classes = _payload["classes"]
    _seq_len = _payload["seq_len"]
    _clean_tracks = _payload.get("clean_tracks", False)
    _in_dim = _payload["feat_dim"]
    _motion_thr = _payload.get("motion_thresh", {})
    _conf_thr = _payload.get("conf_thresh")
    # 臉部／位置特徵已拿掉：用它們訓練出來的模型（use_face / use_loc 為 True）沒法再餵資料
    if _payload.get("use_face") or _payload.get("use_loc") or _in_dim < FEAT_DIM:
        print(f"[警告] 這個模型用了臉部／位置特徵（輸入 {_in_dim} 維），目前已不支援，"
              f"請重新執行 train_model.py 再啟動。")
    else:
        _net = GestureLSTM(
            feat_dim=_in_dim,
            hidden=_payload.get("hidden", 128),
            num_layers=_payload.get("num_layers", 2),
            num_classes=len(_classes),
        )
        _net.load_state_dict(_payload["model_state"])
        _net.to(_device)
        _net.eval()
        _model = _net
        print(f"[模型] 載入成功  裝置：{_device}  輸入 {_in_dim} 維"
              f"  手部濾波：{'啟用' if _clean_tracks else '停用'}"
              f"  詞彙：{_classes}")
except FileNotFoundError:
    print(f"[警告] 找不到 {MODEL_PATH}，請先執行 train_model.py")


def is_ready() -> bool:
    return _model is not None


def default_conf() -> float:
    """模型校準過的信心門檻（沒有就用 DEFAULT_CONF）；網頁「恢復預設」用。"""
    return float(_conf_thr or DEFAULT_CONF)


def get_classes() -> list:
    return list(_classes)


def draw_landmarks(frame_bgr, results):
    """在 BGR 影格上畫出手部骨架。"""
    if results and results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame_bgr, lm, mp_hands.HAND_CONNECTIONS)
    return frame_bgr


# ── 有狀態的即時辨識器 ──────────────────────────
class SignRecognizer:
    """逐幀餵入 RGB 影格，內部維護滑動視窗 + 穩定確認 + 冷卻。

    process() 回傳當前狀態；當某個詞連續穩定出現時，
    state["confirmed"] 會是該詞（否則為 None）。
    """

    def __init__(self, conf_thresh=None, confirm_frames=15, cooldown_frames=30,
                 no_hand_clear=10, motion_thresh=MOTION_THRESH):
        # motion_thresh 0.003：小幅度手勢（謝謝、什麼）中位動作僅 ~0.004，
        # 0.005 會擋掉過半正常樣本；垃圾靜止樣本多在 0.0016 以下，0.003 仍擋得住
        # 參數與 collect_data.py 共用（src/features.py），確保收集/推理偵測行為一致
        self.hands = mp_hands.Hands(**MP_HANDS_KWARGS)
        self.seq_len         = _seq_len
        # 信心門檻：沒指定就用模型校準過的值（見 train_model.py），不同詞彙量的模型信心高低差很多
        self.conf_thresh     = conf_thresh if conf_thresh is not None else (_conf_thr or DEFAULT_CONF)
        self.confirm_frames  = confirm_frames
        self.cooldown_frames = cooldown_frames
        self.no_hand_clear   = no_hand_clear
        self.motion_thresh   = motion_thresh

        self.buffer        = deque(maxlen=self.seq_len)
        self.confirm_word  = ""
        self.confirm_count = 0
        self.cooldown      = 0
        self.no_hand       = 0
        self.idle_frames   = 0   # 連續沒有手的幀數（給上層判斷句子結束）

    def reset(self):
        self.buffer.clear()
        self.confirm_word  = ""
        self.confirm_count = 0
        self.cooldown      = 0

    def process(self, frame_rgb):
        results  = self.hands.process(frame_rgb)
        has_hand = results.multi_hand_landmarks is not None

        if has_hand:
            self.no_hand = 0
            self.idle_frames = 0
            self.buffer.append(extract_two_hands(results))
        else:
            self.no_hand += 1
            self.idle_frames += 1
            if self.no_hand >= self.no_hand_clear:
                self.buffer.clear()
                self.confirm_word  = ""
                self.confirm_count = 0

        raw_word, raw_conf, motion = "", 0.0, 0.0
        top = []          # 這一幀模型最有把握的前幾名 [(詞, 信心)]；沒跑模型就是空的
        if self.cooldown > 0:
            self.cooldown -= 1
        elif _model is not None and has_hand and len(self.buffer) == self.seq_len:
            arr    = np.array(self.buffer)
            motion = hand_motion(arr)
            # 先用最寬鬆的門檻決定要不要跑模型，預測出詞之後再用「該詞自己的門檻」確認：
            # 靜態手勢（誰、是）門檻較低，其餘詞維持 motion_thresh
            loosest = min([self.motion_thresh, *_motion_thr.values()])
            if motion >= loosest:
                # 動作量刻意在濾波「前」計算：門檻 0.003 是用原始資料校準的，
                # 濾波會降低動作量，先濾可能把小動作的手勢（如謝謝）擋掉
                if _clean_tracks:
                    arr = clean_hand_tracks(arr)
                feat = to_model_input(arr)
                if _in_dim > feat.shape[1]:         # 舊模型：後面是補零的臉部欄位
                    feat = np.pad(feat, ((0, 0), (0, _in_dim - feat.shape[1])))
                x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(_device)
                with torch.no_grad():
                    proba = torch.softmax(_model(x), dim=1)[0]
                conf, idx = proba.max(0)
                word = _classes[idx.item()]
                k = min(TOP_K, len(_classes))
                vals, ids = torch.topk(proba, k)
                top = [(_classes[i], round(float(v), 3)) for v, i in zip(vals.tolist(), ids.tolist())]
                need = min(self.motion_thresh, _motion_thr.get(word, self.motion_thresh))
                if conf.item() >= self.conf_thresh and motion >= need:
                    raw_word, raw_conf = word, conf.item()

        # 穩定確認
        if raw_word:
            if raw_word == self.confirm_word:
                self.confirm_count += 1
            else:
                self.confirm_word, self.confirm_count = raw_word, 1
        else:
            self.confirm_word, self.confirm_count = "", 0

        confirmed, confirmed_conf = None, 0.0
        if self.confirm_count >= self.confirm_frames:
            confirmed, confirmed_conf = self.confirm_word, raw_conf
            self.confirm_word, self.confirm_count = "", 0
            self.cooldown = self.cooldown_frames
            self.buffer.clear()

        return {
            "results":        results,
            "has_hand":       has_hand,
            "hand_n":         len(results.multi_hand_landmarks) if has_hand else 0,
            "raw_word":       raw_word,
            "raw_conf":       raw_conf,
            "top":            top,
            "motion":         motion,
            "buffer_pct":     int(len(self.buffer) / self.seq_len * 100),
            "confirm_word":   self.confirm_word,
            "confirm_pct":    int(self.confirm_count / self.confirm_frames * 100),
            "cooldown":       self.cooldown,
            "confirmed":      confirmed,
            "confirmed_conf": confirmed_conf,
            "idle_frames":    self.idle_frames,
        }
