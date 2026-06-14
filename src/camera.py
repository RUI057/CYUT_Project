import mediapipe as mp
import numpy as np
import pickle
import torch
import torch.nn as nn
from collections import deque

mp_hands = mp.solutions.hands
_hands_static  = mp_hands.Hands(static_image_mode=True,  max_num_hands=2,
                                 min_detection_confidence=0.6)
_hands_dynamic = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                 min_detection_confidence=0.7)

# ── LSTM 模型定義
class GestureLSTM(nn.Module):
    def __init__(self, feat_dim, hidden=128, num_layers=2, num_classes=10, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))

# ── 推理裝置
if torch.backends.mps.is_available():
    _device = torch.device("mps")
elif torch.cuda.is_available():
    _device = torch.device("cuda")
else:
    _device = torch.device("cpu")

# ── 載入模型
_model = _classes = _seq_len = None

try:
    with open("data/model.pkl", "rb") as f:
        payload = pickle.load(f)

    _classes  = payload["classes"]
    _seq_len  = payload["seq_len"]
    _feat_dim = payload["feat_dim"]

    net = GestureLSTM(
        feat_dim=_feat_dim,
        hidden=payload.get("hidden", 128),
        num_layers=payload.get("num_layers", 2),
        num_classes=len(_classes)
    )
    net.load_state_dict(payload["model_state"])
    net.to(_device)
    net.eval()
    _model = net
    print(f"[模型] 載入成功  裝置：{_device}  詞彙：{_classes}")

except FileNotFoundError:
    print("[警告] 找不到 data/model.pkl，請先執行 train_model.py")

# ── 滑動窗口緩衝區 ────────────────────────────
_buffer = deque(maxlen=_seq_len if _seq_len else 30)

# ── 特徵擷取 ──────────────────────────────────
def _extract_two_hands(results):
    left  = [0.0] * 63
    right = [0.0] * 63
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handedness in zip(results.multi_hand_landmarks,
                                        results.multi_handedness):
            lbl   = handedness.classification[0].label
            wrist = hand_lms.landmark[0]
            feat  = []
            for lm in hand_lms.landmark:
                feat += [lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]
            if lbl == "Left":  left  = feat
            else:              right = feat
    return left + right

# ── 公開 API ──────────────────────────────────
def has_hand_from_array(frame_rgb: np.ndarray) -> bool:
    result = _hands_static.process(frame_rgb)
    return result.multi_hand_landmarks is not None

def recognize_local_from_array(frame_rgb: np.ndarray) -> str:
    """
    輸入 RGB numpy array，回傳辨識詞彙。
    信心不足或幀數未累積回傳空字串。
    """
    global _buffer
    if _model is None:
        return "？"
    try:
        results = _hands_dynamic.process(frame_rgb)
        _buffer.append(_extract_two_hands(results))

        if len(_buffer) < _seq_len:
            return ""

        x = torch.tensor(np.array(_buffer), dtype=torch.float32)
        x = x.unsqueeze(0).to(_device)          # (1, 30, 126)

        with torch.no_grad():
            logits = _model(x)
            proba  = torch.softmax(logits, dim=1)[0]
            conf, idx = proba.max(0)

        if conf.item() >= 0.75:
            return _classes[idx.item()]
        return ""

    except Exception as e:
        print(f"[辨識錯誤] {e}")
        return ""

def reset_buffer():
    global _buffer
    _buffer.clear()
