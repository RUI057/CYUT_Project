"""攝影機端手語辨識：載入 LSTM 模型、即時推理、穩定確認。

架構與 train_model.py 一致（雙向 LSTM + mean/max 池化），
特徵抽取與尺度正規化共用 src/features，確保訓練/推理一致。
特徵含雙手(126)+臉部非手動標記(36)=162 維；臉部用獨立的 FaceMesh，
手部沿用 mp.solutions.hands，確保與舊資料的手部特徵完全相容。
"""
import os
import sys
import pickle
from collections import deque

import numpy as np
import torch
import mediapipe as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import extract_features, normalize_sequence, FEAT_DIM, HANDS_DIM
from src.model import GestureLSTM

mp_hands = mp.solutions.hands
mp_face  = mp.solutions.face_mesh
mp_draw  = mp.solutions.drawing_utils


# ── 推理裝置（LSTM 定義集中於 src/model.py）──────────────
if torch.backends.mps.is_available():
    _device = torch.device("mps")
elif torch.cuda.is_available():
    _device = torch.device("cuda")
else:
    _device = torch.device("cpu")

# ── 載入模型 ──────────────────────────────────
_model = None
_classes = []
_seq_len = 30

try:
    with open("data/model.pkl", "rb") as f:
        _payload = pickle.load(f)
    _classes = _payload["classes"]
    _seq_len = _payload["seq_len"]
    if _payload["feat_dim"] != FEAT_DIM:
        print(f"[警告] 模型特徵維度 {_payload['feat_dim']} 與目前管線 {FEAT_DIM} 不符"
              f"（已加入臉部偵測），請重新執行 train_model.py 再啟動。")
    else:
        _net = GestureLSTM(
            feat_dim=_payload["feat_dim"],
            hidden=_payload.get("hidden", 128),
            num_layers=_payload.get("num_layers", 2),
            num_classes=len(_classes),
        )
        _net.load_state_dict(_payload["model_state"])
        _net.to(_device)
        _net.eval()
        _model = _net
        print(f"[模型] 載入成功  裝置：{_device}  詞彙：{_classes}")
except FileNotFoundError:
    print("[警告] 找不到 data/model.pkl，請先執行 train_model.py")


def is_ready() -> bool:
    return _model is not None


def get_classes() -> list:
    return list(_classes)


def draw_landmarks(frame_bgr, results, face_results=None):
    """在 BGR 影格上畫出手部骨架；有臉部結果時一併畫臉部輪廓。"""
    if results and results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame_bgr, lm, mp_hands.HAND_CONNECTIONS)
    if face_results and getattr(face_results, "multi_face_landmarks", None):
        for lm in face_results.multi_face_landmarks:
            mp_draw.draw_landmarks(frame_bgr, lm, mp_face.FACEMESH_CONTOURS)
    return frame_bgr


# ── 有狀態的即時辨識器 ──────────────────────────
class SignRecognizer:
    """逐幀餵入 RGB 影格，內部維護滑動視窗 + 穩定確認 + 冷卻。

    process() 回傳當前狀態；當某個詞連續穩定出現時，
    state["confirmed"] 會是該詞（否則為 None）。
    """

    def __init__(self, conf_thresh=0.80, confirm_frames=15, cooldown_frames=30,
                 no_hand_clear=10, motion_thresh=0.003):
        # motion_thresh 0.003：小幅度手勢（謝謝、什麼）中位動作僅 ~0.004，
        # 0.005 會擋掉過半正常樣本；垃圾靜止樣本多在 0.0016 以下，0.003 仍擋得住
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                    min_detection_confidence=0.7,
                                    min_tracking_confidence=0.5)
        # 臉部非手動標記（眉/眼/嘴）；獨立於手部模型，手部特徵與舊資料相容
        self.face = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                                     refine_landmarks=False,
                                     min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)
        self.seq_len         = _seq_len
        self.conf_thresh     = conf_thresh
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
        face_res = self.face.process(frame_rgb)
        has_hand = results.multi_hand_landmarks is not None
        has_face = bool(getattr(face_res, "multi_face_landmarks", None))

        if has_hand:
            self.no_hand = 0
            self.idle_frames = 0
            self.buffer.append(extract_features(results, face_res))
        else:
            self.no_hand += 1
            self.idle_frames += 1
            if self.no_hand >= self.no_hand_clear:
                self.buffer.clear()
                self.confirm_word  = ""
                self.confirm_count = 0

        raw_word, raw_conf, motion = "", 0.0, 0.0
        if self.cooldown > 0:
            self.cooldown -= 1
        elif _model is not None and has_hand and len(self.buffer) == self.seq_len:
            arr    = np.array(self.buffer)
            # 只看手部動作量：臉部表情(眨眼/嘴動)不該影響「手有沒有在比」的判斷
            motion = float(np.std(arr[:, :HANDS_DIM], axis=0).mean())
            if motion >= self.motion_thresh:
                x = torch.tensor(normalize_sequence(arr),
                                 dtype=torch.float32).unsqueeze(0).to(_device)
                with torch.no_grad():
                    proba = torch.softmax(_model(x), dim=1)[0]
                conf, idx = proba.max(0)
                if conf.item() >= self.conf_thresh:
                    raw_word, raw_conf = _classes[idx.item()], conf.item()

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
            "face_results":   face_res,
            "has_hand":       has_hand,
            "has_face":       has_face,
            "hand_n":         len(results.multi_hand_landmarks) if has_hand else 0,
            "raw_word":       raw_word,
            "raw_conf":       raw_conf,
            "motion":         motion,
            "buffer_pct":     int(len(self.buffer) / self.seq_len * 100),
            "confirm_word":   self.confirm_word,
            "confirm_pct":    int(self.confirm_count / self.confirm_frames * 100),
            "cooldown":       self.cooldown,
            "confirmed":      confirmed,
            "confirmed_conf": confirmed_conf,
            "idle_frames":    self.idle_frames,
        }
