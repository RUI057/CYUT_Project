import cv2
import mediapipe as mp
import numpy as np
import pickle
import sys
import torch
import torch.nn as nn
from collections import deque
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from src.font_utils import get_font, put_text
from src.features import extract_two_hands, normalize_sequence

# ── LSTM 定義（與 train_model.py 一致）────────
class GestureLSTM(nn.Module):
    def __init__(self, feat_dim, hidden=128, num_layers=2, num_classes=10, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 4, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, num_classes)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        feat = torch.cat([out.mean(1), out.max(1).values], dim=1)
        return self.fc(self.dropout(feat))

# ── 載入模型 ──────────────────────────────────
try:
    with open("data/model.pkl", "rb") as f:
        payload = pickle.load(f)
    classes   = payload["classes"]
    seq_len   = payload["seq_len"]
    feat_dim  = payload["feat_dim"]

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = GestureLSTM(feat_dim=feat_dim, num_classes=len(classes))
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    print(f"[OK] 模型載入  裝置：{device}")
    print(f"[OK] 詞彙：{classes}")
except FileNotFoundError:
    print("[錯誤] 找不到 data/model.pkl")
    sys.exit(1)

# ── MediaPipe ─────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                          min_detection_confidence=0.7)

font_large  = get_font(44)
font_medium = get_font(28)
font_small  = get_font(20)

# 特徵擷取改用 src/features.extract_two_hands（與 collect/train 共用，確保一致）

# ── 參數 ──────────────────────────────────────
CONF_THRESH     = 0.80   # 信心度門檻
CONFIRM_FRAMES  = 20     # 同一詞連續幾幀才確認
COOLDOWN_FRAMES = 20     # 確認後冷卻幾幀
NO_HAND_CLEAR   = 10     # 手消失幾幀後清空 buffer
MOTION_THRESH   = 0.005  # 動作門檻，調高=更難觸發，調低=更容易觸發

# ── 開啟攝影機 ────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    for i in range(1, 4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            break

buffer         = deque(maxlen=seq_len)
confirm_word   = ""
confirm_count  = 0
cooldown       = 0
confirmed_word = ""
confirmed_conf = 0.0
result_history = deque(maxlen=6)
no_hand_count  = 0
motion_score   = 0.0   # 即時動作分數

print(f"\n[動作門檻] MOTION_THRESH = {MOTION_THRESH}")
print("[操作]  C=清除  Q=離開\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape

    hand_detected = results.multi_hand_landmarks is not None

    if hand_detected:
        for lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
        no_hand_count = 0
        buffer.append(extract_two_hands(results))
    else:
        no_hand_count += 1
        # 手消失超過 NO_HAND_CLEAR 幀 → 清空 buffer 和候選詞
        if no_hand_count >= NO_HAND_CLEAR:
            buffer.clear()
            confirm_word  = ""
            confirm_count = 0

    # ── PyTorch 推理 ──────────────────────────
    raw_pred, raw_conf = "", 0.0
    if cooldown > 0:
        cooldown -= 1
    elif hand_detected and len(buffer) == seq_len:
        
        # 計算動作幅度
        motion_score = float(np.std(np.array(buffer), axis=0).mean())

        if motion_score < MOTION_THRESH:
            # 變化量太小，代表手是靜止的，跳過預測
            pass 
        else:
            # 餵模型前做尺度正規化（與訓練一致）
            seq = normalize_sequence(np.array(buffer))
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                proba = torch.softmax(model(x), dim=1)[0]
            conf, idx = proba.max(0)
            if conf.item() >= CONF_THRESH:
                raw_pred = classes[idx.item()]
                raw_conf = conf.item()
                
    # ── 穩定確認邏輯 ──────────────────────────
    if raw_pred:
        if raw_pred == confirm_word:
            confirm_count += 1
        else:
            confirm_word  = raw_pred
            confirm_count = 1
    else:
        confirm_word  = ""
        confirm_count = 0

    if confirm_count >= CONFIRM_FRAMES:
        confirmed_word = confirm_word
        confirmed_conf = raw_conf
        result_history.append((confirmed_word, confirmed_conf))
        print(f"[確認] {confirmed_word}  {confirmed_conf:.0%}")
        confirm_word  = ""
        confirm_count = 0
        cooldown      = COOLDOWN_FRAMES
        buffer.clear()   # 清空舊幀，避免殘留誤判

    # ── 畫面 ──────────────────────────────────
    buf_pct = int(len(buffer) / seq_len * 100)

    # Buffer 進度條
    cv2.rectangle(frame, (10, h-28), (w-10, h-10), (40,40,40), -1)
    bw = int((w-20) * buf_pct / 100)
    cv2.rectangle(frame, (10, h-28), (10+bw, h-10),
                  (0,180,80) if buf_pct==100 else (0,100,200), -1)
    cv2.putText(frame, f"Buffer {buf_pct}%", (15, h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    # 確認進度條
    if confirm_word:
        prog = int(confirm_count / CONFIRM_FRAMES * 100)
        cv2.rectangle(frame, (10, h-52), (w-10, h-34), (40,40,40), -1)
        pw = int((w-20) * prog / 100)
        cv2.rectangle(frame, (10, h-52), (10+pw, h-34), (0,200,160), -1)
        frame = put_text(frame, f"確認中：{confirm_word} {prog}%",
                         (15, h-56), font_small, (0,220,180))

    if cooldown > 0:
        frame = put_text(frame, f"冷卻 {cooldown}",
                         (10, h-78), font_small, (180,180,80))

    hc     = (0,220,80) if hand_detected else (80,80,200)
    hand_n = len(results.multi_hand_landmarks) if hand_detected else 0
    cv2.putText(frame, f"Hand:{hand_n}", (10,32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, hc, 2)

    if confirmed_word:
        frame = put_text(frame, confirmed_word,         (10,48),  font_large,  (0,220,80))
        frame = put_text(frame, f"{confirmed_conf:.0%}", (10,100), font_small, (160,255,160))
    else:
        frame = put_text(frame, "比手語等待辨識...", (10,48), font_medium, (130,130,130))

    frame = put_text(frame, "辨識紀錄", (w-190,10), font_small, (180,180,180))
    for i, (word, conf) in enumerate(reversed(result_history)):
        b = max(80, 240 - i*35)
        frame = put_text(frame, f"{word}  {conf:.0%}",
                         (w-190, 36+i*30), font_small, (b,b,b))

    # 動作分數（幫助校準 MOTION_THRESH）
    mc = (0, 220, 80) if motion_score >= MOTION_THRESH else (80, 80, 200)
    cv2.putText(frame, f"Motion:{motion_score:.4f}/{MOTION_THRESH}",
                (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mc, 1)
    cv2.putText(frame, "C=clear  Q=quit", (10, h-44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,100,100), 1)

    cv2.imshow("手語模型測試 (LSTM)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        result_history.clear()
        confirmed_word = ""
        confirm_word   = ""
        confirm_count  = 0
        buffer.clear()
        print("[清除]")

cap.release()
cv2.destroyAllWindows()
