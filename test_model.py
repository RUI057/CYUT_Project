import cv2
import mediapipe as mp
import numpy as np
import pickle
import sys
from collections import deque
from PIL import Image, ImageDraw, ImageFont

# ── 載入模型 ──────────────────────────────────
try:
    with open("data/model.pkl", "rb") as f:
        payload = pickle.load(f)
    model   = payload["model"]
    labels  = payload["labels"]
    seq_len = payload["seq_len"]
    print(f"[OK] 模型載入，詞彙：{labels}")
except FileNotFoundError:
    print("[錯誤] 找不到 data/model.pkl，請先執行 train_model.py")
    sys.exit(1)

# ── MediaPipe ─────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                          min_detection_confidence=0.7)

# ── 字型 ──────────────────────────────────────
mac_font_paths = [
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc"
]

font_large = font_medium = font_small = None
for path in mac_font_paths:
    try:
        font_large  = ImageFont.truetype(path, 44)
        font_medium = ImageFont.truetype(path, 28)
        font_small  = ImageFont.truetype(path, 20)
        print(f"[OK] 成功載入中文字型：{path}")
        break
    except Exception:
        continue


if font_large is None:
    print("[警告] 找不到支援中文的字型，畫面可能會報錯！")
    font_large = font_medium = font_small = ImageFont.load_default()

def put_text(frame, text, pos, font, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ── 特徵擷取 ──────────────────────────────────
def extract_two_hands(results):
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
            if lbl == "Left":
                left = feat
            else:
                right = feat
    return left + right

# ── 參數設定 ──────────────────────────────────
CONF_THRESH    = 0.75   # 信心度門檻
CONFIRM_FRAMES = 20     # 同一詞連續出現幾幀才確認
COOLDOWN_FRAMES = 30    # 確認後冷卻幾幀，防止重複

# ── 狀態變數 ──────────────────────────────────
cap           = cv2.VideoCapture(0)
if not cap.isOpened():
    for i in range(1, 4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            break

buffer         = deque(maxlen=seq_len)
confirm_word   = ""      # 目前累積中的候選詞
confirm_count  = 0       # 候選詞連續出現幾幀
cooldown       = 0       # 冷卻計數
confirmed_word = ""      # 已確認的詞
confirmed_conf = 0.0
result_history = deque(maxlen=6)

print("\n[操作說明]  C=清除  Q=離開\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape

    # 畫手部骨架
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    # 累積特徵
    buffer.append(extract_two_hands(results))

    # 冷卻中不做預測
    if cooldown > 0:
        cooldown -= 1
        raw_pred = ""
        raw_conf = 0.0
    elif len(buffer) == seq_len:
        X        = np.array(buffer).flatten().reshape(1, -1)
        proba    = model.predict_proba(X)[0]
        idx      = proba.argmax()
        raw_conf = proba[idx]
        raw_pred = model.classes_[idx] if raw_conf >= CONF_THRESH else ""
    else:
        raw_pred = ""
        raw_conf = 0.0

    # ── 穩定確認邏輯 ──────────────────────────
    if raw_pred:
        if raw_pred == confirm_word:
            confirm_count += 1
        else:
            # 換詞了，重新從 1 開始累積
            confirm_word  = raw_pred
            confirm_count = 1
    else:
        # 沒偵測到就重置候選
        confirm_word  = ""
        confirm_count = 0

    # 累積夠了 → 確認
    if confirm_count >= CONFIRM_FRAMES:
        confirmed_word = confirm_word
        confirmed_conf = raw_conf
        result_history.append((confirmed_word, confirmed_conf))
        print(f"[確認] {confirmed_word}  {confirmed_conf:.0%}")
        confirm_word  = ""
        confirm_count = 0
        cooldown      = COOLDOWN_FRAMES

    # ── 畫面顯示 ──────────────────────────────
    hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
    buf_pct    = int(len(buffer) / seq_len * 100)

    # Buffer 進度條（底部）
    cv2.rectangle(frame, (10, h - 28), (w - 10, h - 10), (40, 40, 40), -1)
    fill_w = int((w - 20) * buf_pct / 100)
    bar_color = (0, 180, 80) if buf_pct == 100 else (0, 100, 200)
    cv2.rectangle(frame, (10, h - 28), (10 + fill_w, h - 10), bar_color, -1)
    cv2.putText(frame, f"Buffer {buf_pct}%", (15, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # 確認進度條（中間偏下）
    if confirm_word:
        prog_pct = int(confirm_count / CONFIRM_FRAMES * 100)
        cv2.rectangle(frame, (10, h - 52), (w - 10, h - 34), (40, 40, 40), -1)
        prog_w = int((w - 20) * prog_pct / 100)
        cv2.rectangle(frame, (10, h - 52), (10 + prog_w, h - 34), (0, 220, 180), -1)
        frame = put_text(frame, f"確認中：{confirm_word}  {prog_pct}%",
                         (15, h - 53), font_small, (0, 220, 180))

    # 冷卻提示
    if cooldown > 0:
        frame = put_text(frame, f"冷卻中 {cooldown}", (10, h - 80),
                         font_small, (180, 180, 80))

    # 有手 / 無手
    hc = (0, 220, 80) if hand_count > 0 else (80, 80, 200)
    cv2.putText(frame, f"Hand: {hand_count}", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, hc, 2)

    # 已確認詞彙（大字）
    if confirmed_word:
        frame = put_text(frame, confirmed_word, (10, 48),  font_large,  (0, 220, 80))
        frame = put_text(frame, f"{confirmed_conf:.0%}",   (10, 100), font_small, (160, 255, 160))
    else:
        frame = put_text(frame, "比手語等待辨識...", (10, 48), font_medium, (130, 130, 130))

    # 右側辨識紀錄
    frame = put_text(frame, "辨識紀錄", (w - 190, 10), font_small, (180, 180, 180))
    for i, (word, conf) in enumerate(reversed(result_history)):
        brightness = max(80, 240 - i * 35)
        c = (brightness, brightness, brightness)
        frame = put_text(frame, f"{word}  {conf:.0%}",
                         (w - 190, 36 + i * 30), font_small, c)

    # 操作說明
    cv2.putText(frame, "C=clear  Q=quit", (10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

    cv2.imshow("手語模型測試", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        result_history.clear()
        confirmed_word = ""
        confirm_word   = ""
        confirm_count  = 0
        buffer.clear()
        print("[清除] 紀錄已清除")

cap.release()
cv2.destroyAllWindows()
