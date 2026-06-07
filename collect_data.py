import cv2
import mediapipe as mp
import numpy as np
import os
import sys
from PIL import Image, ImageDraw, ImageFont  # noqa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.vocab import get_all_labels

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                          min_detection_confidence=0.7)

from src.font_utils import get_font, put_text
font_large  = get_font(50)
font_medium = get_font(40)
font_small  = get_font(40)

# ── 設定 ──────────────────────────────────────
GESTURES         = get_all_labels()
SEQUENCE_LENGTH  = 30   # 每筆幾幀
SAMPLES_PER_CLASS = 80  # 每個詞彙幾筆
HOLD_FRAMES      = 15   # 手穩定幾幀後自動開始錄製
COOLDOWN_FRAMES  = 20   # 每筆錄完後冷卻幾幀（避免連續誤觸）
DATA_DIR         = "dynamic_dataset"

os.makedirs(DATA_DIR, exist_ok=True)
for g in GESTURES:
    os.makedirs(os.path.join(DATA_DIR, g), exist_ok=True)

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

# ── 攝影機 
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    for i in range(1, 4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            break

# ── 狀態 
current_gesture = 0
state           = "waiting"   # waiting → holding → recording → cooldown
sequence        = []
hold_count      = 0
cooldown_count  = 0
paused          = False

print("[詞彙清單]")
for i, g in enumerate(GESTURES):
    existing = len(os.listdir(os.path.join(DATA_DIR, g)))
    status   = "OK" if existing >= SAMPLES_PER_CLASS else f"{existing}/{SAMPLES_PER_CLASS}"
    print(f"  {i:2d} = {g:6s}  [{status}]")
print("\n[操作] N=下一個  P=上一個  R=重置此詞彙  D=刪除上一個  S=暫停/繼續偵測  Q=離開")
print("[提示] 按 S 可暫停偵測避免誤觸，按 D 可刪除上一筆已儲存資料\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape

    label    = GESTURES[current_gesture]
    existing = len(os.listdir(os.path.join(DATA_DIR, label)))
    has_hand = results.multi_hand_landmarks is not None

    # 畫手部骨架
    if has_hand:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    # ── 狀態機
    if paused:
        hold_count = 0

    elif existing >= SAMPLES_PER_CLASS:
        state = "done"

    elif state == "waiting":
        if has_hand:
            hold_count += 1
            if hold_count >= HOLD_FRAMES:
                state      = "recording"
                sequence   = []
                hold_count = 0
                print(f"[錄製] {label}  第 {existing + 1} 筆")
        else:
            hold_count = 0

    elif state == "recording":
        feat = extract_two_hands(results)
        sequence.append(feat)
        if len(sequence) == SEQUENCE_LENGTH:
            arr       = np.array(sequence)
            save_path = os.path.join(DATA_DIR, label, f"{existing}.npy")
            np.save(save_path, arr)
            existing += 1
            print(f"[儲存] {label}  {existing}/{SAMPLES_PER_CLASS}")
            state          = "cooldown"
            cooldown_count = COOLDOWN_FRAMES
            sequence       = []

    elif state == "cooldown":
        cooldown_count -= 1
        if cooldown_count <= 0:
            state = "waiting" if existing < SAMPLES_PER_CLASS else "done"

    # ── 畫面顯示 
    # 詞彙索引與名稱
    idx_text = f"[{current_gesture + 1}/{len(GESTURES)}]"
    frame = put_text(frame, f"{idx_text} {label}", (10, 10), font_large,
                     (0, 220, 80) if state == "recording" else (255, 255, 255))

    # 進度
    frame = put_text(frame, f"已收集：{existing} / {SAMPLES_PER_CLASS}",
                     (10, 60), font_medium, (180, 180, 180))

    # 狀態
    state_info = {
        "waiting":   (f"等待手部出現... ({hold_count}/{HOLD_FRAMES})", (150, 150, 150)),
        "recording": (f"錄製中  {len(sequence)}/{SEQUENCE_LENGTH} 幀",  (0, 220, 80)),
        "cooldown":  (f"冷卻中  {cooldown_count}",                      (0, 160, 255)),
        "done":      ("此詞彙已完成！按 N 換下一個",                     (0, 220, 80)),
        "paused":    ("暫停偵測，按 S 繼續",                         (220, 180, 0)),
    }
    frame = put_text(frame, state_info["paused"][0] if paused else state_info[state][0],
                     (10, 95), font_medium, state_info["paused"][1] if paused else state_info[state][1])

    # 有無偵測到手
    hand_text  = f"偵測到 {len(results.multi_hand_landmarks)} 隻手" if has_hand else "未偵測到手"
    hand_color = (100, 220, 255) if has_hand else (80, 80, 200)
    frame = put_text(frame, hand_text, (10, 130), font_small, hand_color)

    # 進度條
    bar_w = w - 20
    cv2.rectangle(frame, (10, h - 25), (10 + bar_w, h - 8), (50, 50, 50), -1)
    filled = int(bar_w * min(existing, SAMPLES_PER_CLASS) / SAMPLES_PER_CLASS)
    cv2.rectangle(frame, (10, h - 25), (10 + filled, h - 8), (0, 180, 80), -1)

    # 操作提示
    frame = put_text(frame, "N=下一個  P=上一個  R=重置  D=刪除上一個  S=暫停/繼續  Q=離開",
                     (10, h - 45), font_small, (120, 120, 120))

    # 錄製中閃爍紅點
    if state == "recording":
        cv2.circle(frame, (w - 30, 25), 10, (0, 0, 220), -1)

    cv2.imshow("手語資料收集", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('n'):
        current_gesture = (current_gesture + 1) % len(GESTURES)
        state = "waiting"
        sequence = []
        hold_count = 0
        paused = True  
        print(f"[切換] -> {GESTURES[current_gesture]}")
    elif key == ord('p'):
        current_gesture = (current_gesture - 1) % len(GESTURES)
        state = "waiting"
        sequence = []
        hold_count = 0
        paused = True  
        print(f"[切換] -> {GESTURES[current_gesture]}")
    elif key == ord('r'):
        # 重置此詞彙的資料
        folder = os.path.join(DATA_DIR, label)
        for f in os.listdir(folder):
            if f.endswith(".npy"):
                os.remove(os.path.join(folder, f))
        state = "waiting"
        sequence = []
        hold_count = 0
        paused = False
        print(f"[重置] {label} 的資料已清除")
    elif key == ord('d'):
        folder = os.path.join(DATA_DIR, label)
        samples = [f for f in os.listdir(folder) if f.endswith('.npy')]
        if samples:
            samples.sort(key=lambda x: int(os.path.splitext(x)[0]))
            last_file = samples[-1]
            os.remove(os.path.join(folder, last_file))
            state = "waiting"
            sequence = []
            hold_count = 0
            print(f"[刪除] 已移除 {label} 的上一筆資料：{last_file}")
        else:
            print(f"[刪除] {label} 尚無資料可刪除")
    elif key == ord('s'):
        paused = not paused
        if paused:
            state = "waiting"
            sequence = []
            hold_count = 0
            print("暫停")
        else:
            print("恢復")

cap.release()
cv2.destroyAllWindows()

# 最終統計
print("\n[最終收集統計]")
all_done = True
for g in GESTURES:
    n      = len(os.listdir(os.path.join(DATA_DIR, g)))
    bar    = "#" * (n // 4)
    status = "OK" if n >= SAMPLES_PER_CLASS else "--"
    if n < SAMPLES_PER_CLASS:
        all_done = False
    print(f"  [{status}] {g:6s}  {bar} {n}/{SAMPLES_PER_CLASS}")

if all_done:
    print("\n全部詞彙收集完成")
else:
    print("\n還沒完成")
