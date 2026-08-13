"""即時辨識測試（OpenCV 視窗）。

辨識邏輯共用 src/camera.SignRecognizer（與網頁 app.py 同一套），
本檔只負責畫面顯示與按鍵操作。
"""
import os
import sys
from collections import deque

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import camera
from src.font_utils import get_font, put_text

if not camera.is_ready():
    print("[錯誤] 找不到 data/model.pkl，請先執行 train_model.py")
    sys.exit(1)

print(f"[OK] 詞彙：{camera.get_classes()}")

rec = camera.SignRecognizer()   # 預設門檻，與 app.py 一致

font_large  = get_font(44)
font_medium = get_font(28)
font_small  = get_font(20)

# ── 開啟攝影機 ────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    for i in range(1, 4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            break

confirmed_word = ""
confirmed_conf = 0.0
result_history = deque(maxlen=6)

print("[操作]  C=清除  Q=離開\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    state = rec.process(rgb)
    camera.draw_landmarks(frame, state["results"], state.get("face_results"))

    if state["confirmed"]:
        confirmed_word = state["confirmed"]
        confirmed_conf = state["confirmed_conf"]
        result_history.append((confirmed_word, confirmed_conf))
        print(f"[確認] {confirmed_word}  {confirmed_conf:.0%}")

    # ── 畫面 ──────────────────────────────────
    buf_pct = state["buffer_pct"]

    # Buffer 進度條
    cv2.rectangle(frame, (10, h-28), (w-10, h-10), (40, 40, 40), -1)
    bw = int((w-20) * buf_pct / 100)
    cv2.rectangle(frame, (10, h-28), (10+bw, h-10),
                  (0, 180, 80) if buf_pct == 100 else (0, 100, 200), -1)
    cv2.putText(frame, f"Buffer {buf_pct}%", (15, h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # 確認進度條
    if state["confirm_word"]:
        prog = state["confirm_pct"]
        cv2.rectangle(frame, (10, h-52), (w-10, h-34), (40, 40, 40), -1)
        pw = int((w-20) * prog / 100)
        cv2.rectangle(frame, (10, h-52), (10+pw, h-34), (0, 200, 160), -1)
        frame = put_text(frame, f"確認中：{state['confirm_word']} {prog}%",
                         (15, h-56), font_small, (0, 220, 180))

    if state["cooldown"] > 0:
        frame = put_text(frame, f"冷卻 {state['cooldown']}",
                         (10, h-78), font_small, (180, 180, 80))

    hc = (0, 220, 80) if state["has_hand"] else (80, 80, 200)
    cv2.putText(frame, f"Hand:{state['hand_n']}", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, hc, 2)

    if confirmed_word:
        frame = put_text(frame, confirmed_word,           (10, 48),  font_large, (0, 220, 80))
        frame = put_text(frame, f"{confirmed_conf:.0%}",  (10, 100), font_small, (160, 255, 160))
    else:
        frame = put_text(frame, "比手語等待辨識...", (10, 48), font_medium, (130, 130, 130))

    frame = put_text(frame, "辨識紀錄", (w-190, 10), font_small, (180, 180, 180))
    for i, (word, conf) in enumerate(reversed(result_history)):
        b = max(80, 240 - i*35)
        frame = put_text(frame, f"{word}  {conf:.0%}",
                         (w-190, 36+i*30), font_small, (b, b, b))

    # 動作分數（幫助校準）
    mc = (0, 220, 80) if state["motion"] >= rec.motion_thresh else (80, 80, 200)
    cv2.putText(frame, f"Motion:{state['motion']:.4f}/{rec.motion_thresh}",
                (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mc, 1)
    cv2.putText(frame, "C=clear  Q=quit", (10, h-44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

    cv2.imshow("手語模型測試 (LSTM)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        result_history.clear()
        confirmed_word = ""
        rec.reset()
        print("[清除]")

cap.release()
cv2.destroyAllWindows()
