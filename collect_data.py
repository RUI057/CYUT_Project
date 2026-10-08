import argparse
import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.vocab import get_all_labels, get_hint
from src.features import (extract_features, extract_position, longest_hand_gap, BASE_DIM, GAP_MAX,
                          MP_HANDS_KWARGS, MP_FACE_KWARGS)
from src.video import open_camera, camera_help, disable_ime, read_key
from collections import Counter, deque

disable_ime()   # Windows：避免注音輸入法攔截 N、Q 等按鍵（須在開視窗前）

mp_hands = mp.solutions.hands
mp_face  = mp.solutions.face_mesh
mp_draw  = mp.solutions.drawing_utils
# 參數與推理端 src/camera.py 共用（src/features.py），確保收集/推理偵測行為一致
hands    = mp_hands.Hands(**MP_HANDS_KWARGS)
# 臉部非手動標記（眉/眼/嘴）；與手部分開跑，手部特徵與舊資料相容
face     = mp_face.FaceMesh(**MP_FACE_KWARGS)

from src.font_utils import get_font, put_texts
font_large  = get_font(50)
font_medium = get_font(40)
font_small  = get_font(40)

_ap = argparse.ArgumentParser(description="收集手語資料")
_ap.add_argument("--word", help="從哪個詞開始收集（預設：第一個還沒收滿的詞）")
_args = _ap.parse_args()

# ── 設定
GESTURES         = get_all_labels()
SEQUENCE_LENGTH  = 30   # 每筆幾幀
SAMPLES_PER_CLASS = 200  # 每個詞彙幾筆s
HOLD_FRAMES      = 15   # 手穩定幾幀後進入倒數
COUNTDOWN_FRAMES = 12   # 倒數總幀數
COOLDOWN_FRAMES  = 8   # 每筆錄完後冷卻幾幀
NO_HAND_GRACE    = 5   # 連續幾幀沒偵測到手才算「手不見」（MediaPipe 偶爾掉 1-2 幀）
DISPLAY_SMOOTH   = 7   # 畫面顯示的手數取最近幾幀的多數，避免「2 隻手」一閃一閃
MIN_FACE_PCT     = 0.5 # 一筆裡有臉的幀少於這個比例 → 提醒（位置以臉為基準，沒臉就沒位置）
LOW_FPS          = 20  # 幀率低於這個就用紅字標示：兩幀之間手移動太遠，快速手勢容易跟丟
DATA_DIR         = "dynamic_dataset"

# 畫面最上面兩行（詞彙、已收集）用最顯眼的亮色 + 黑色描邊；顏色是 RGB
LABEL_COLOR      = (255, 230, 0)    # 詞彙：亮黃
LABEL_REC_COLOR  = (0, 255, 110)    # 詞彙（錄製中）：亮綠
COUNT_COLOR      = (0, 240, 255)    # 已收集：亮青
KEYS_COLOR       = (255, 110, 255)  # 下方按鍵提示（N、M、J…）：亮粉紫（黃/青/綠/橘已各有用途）
OUTLINE          = 3                # 描邊粗細（像素）

# 本次收集的 session 標記，寫進檔名 → 訓練時可做「同次錄製不跨組」的誠實切分
SESSION = datetime.now().strftime("%Y%m%d_%H%M%S")

os.makedirs(DATA_DIR, exist_ok=True)
for g in GESTURES:
    os.makedirs(os.path.join(DATA_DIR, g), exist_ok=True)

# extract_two_hands 改用 src/features（與 train/test 共用，確保特徵格式一致）

# ── 攝影機（Windows 優先 DirectShow，見 src/video.py）
cap = open_camera()
if cap is None:
    print("[錯誤] " + camera_help())
    sys.exit(1)

# ── 決定從哪個詞開始
def _count(label):
    folder = os.path.join(DATA_DIR, label)
    return len([f for f in os.listdir(folder) if f.endswith(".npy")]) if os.path.isdir(folder) else 0

if _args.word:
    if _args.word not in GESTURES:
        print(f"[錯誤] 詞彙表裡沒有「{_args.word}」")
        sys.exit(1)
    current_gesture = GESTURES.index(_args.word)
else:
    # 預設從第一個還沒收滿的詞開始
    current_gesture = next((i for i, g in enumerate(GESTURES)
                            if _count(g) < SAMPLES_PER_CLASS), 0)

# ── 狀態
state           = "waiting"   # waiting → countdown → recording → cooldown
sequence        = []
hold_count      = 0
miss_count      = 0                          # 連續沒偵測到手的幀數
recent_hands    = deque(maxlen=DISPLAY_SMOOTH)  # 最近幾幀偵測到的手數
countdown_count = 0
cooldown_count  = 0
paused          = False
sample_warn     = ""                         # 上一筆的品質提醒：臉入鏡不足、手中途跟丟（顯示在畫面上）
frame_times     = deque(maxlen=30)           # 最近 30 幀的時間，算實際幀率
print(f"[Session] {SESSION}")

print("\nN=下一個  M=上一個  J=跳到指定詞彙  D=刪除上一個  S=暫停/繼續  Q=離開")
print("按 S 可暫停偵測避免誤觸，按 D 可刪除上一筆已儲存資料\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_times.append(time.perf_counter())
    fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0.0
    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results  = hands.process(rgb)
    face_res = face.process(rgb)
    h, w, _ = frame.shape

    label    = GESTURES[current_gesture]
    existing = len(os.listdir(os.path.join(DATA_DIR, label)))
    has_hand = results.multi_hand_landmarks is not None
    has_face = bool(getattr(face_res, "multi_face_landmarks", None))
    n_hands  = len(results.multi_hand_landmarks) if has_hand else 0

    # 短暫掉手不算手不見：MediaPipe 常掉 1-2 幀，原本一掉就把倒數整個打斷
    miss_count = 0 if has_hand else miss_count + 1
    hand_ok    = miss_count < NO_HAND_GRACE
    recent_hands.append(n_hands)
    shown_n    = Counter(recent_hands).most_common(1)[0][0]

    # 畫手部骨架
    if has_hand:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
    # 畫臉部輪廓
    if has_face:
        mp_draw.draw_landmarks(frame, face_res.multi_face_landmarks[0],
                               mp_face.FACEMESH_CONTOURS)
    # 畫位置：鼻尖 → 手腕連線，就是模型看到的「手在臉的哪裡」
    pos = extract_position(results, face_res)
    if has_face:
        nose = (int(pos[4] * w), int(pos[5] * h))
        for k in (0, 1):
            if pos[2 * k] or pos[2 * k + 1]:
                cv2.line(frame, nose, (int(pos[2 * k] * w), int(pos[2 * k + 1] * h)),
                         (0, 200, 255), 2)
        cv2.circle(frame, nose, 6, (255, 255, 255), -1)

    # ── 狀態機
    if paused:
        hold_count = 0

    elif existing >= SAMPLES_PER_CLASS:
        state = "done"

    elif state == "waiting":
        if not hand_ok:
            hold_count = 0
        elif has_hand:                       # 容忍期內的掉幀：不累積也不歸零
            hold_count += 1
            if hold_count >= HOLD_FRAMES:
                state           = "countdown"
                countdown_count = COUNTDOWN_FRAMES
                hold_count      = 0

    elif state == "countdown":
        # 倒數期間手必須在；連續掉手超過容忍幀數才退回等待
        if not hand_ok:
            state = "waiting"
            hold_count = 0
        else:
            countdown_count -= 1
            if countdown_count <= 0:
                state    = "recording"
                sequence = []
                print(f"[錄製] {label}  第 {existing + 1} 筆")

    elif state == "recording":
        feat = extract_features(results, face_res)
        sequence.append(feat)
        if len(sequence) == SEQUENCE_LENGTH:
            arr       = np.array(sequence)
            save_path = os.path.join(DATA_DIR, label, f"{SESSION}_{existing}.npy")
            np.save(save_path, arr)
            existing += 1
            # 品質檢查：有臉的幀太少 → 缺位置資訊；手中途跟丟太久 → 濾波補不回來
            face_pct = float((np.abs(arr[:, BASE_DIM + 4:BASE_DIM + 6]).sum(1) > 1e-6).mean())
            gap = longest_hand_gap(arr)
            if gap > GAP_MAX:
                sample_warn = (f"上一筆手連續 {gap} 幀沒抓到（動作太快或光線不足）"
                               f"→ 可按 D 刪除重錄，放慢一點或把燈打亮")
            elif face_pct < MIN_FACE_PCT:
                sample_warn = f"上一筆只有 {face_pct:.0%} 的幀有臉 → 缺位置資訊，可按 D 刪除重錄"
            else:
                sample_warn = ""
            print(f"[儲存] {label}  {existing}/{SAMPLES_PER_CLASS}"
                  + (f"  ⚠️ {sample_warn}" if sample_warn else ""))
            state          = "cooldown"
            cooldown_count = COOLDOWN_FRAMES
            sequence       = []

    elif state == "cooldown":
        cooldown_count -= 1
        if cooldown_count <= 0:
            state = "waiting" if existing < SAMPLES_PER_CLASS else "done"

    # ── 畫面顯示（中文字先收集起來，最後一次畫完：每段各畫一次會讓幀率掉一半以上）
    texts = []
    # 詞彙索引與名稱
    idx_text = f"[{current_gesture + 1}/{len(GESTURES)}]"
    texts.append((f"{idx_text} {label}", (10, 10), font_large,
                  LABEL_REC_COLOR if state == "recording" else LABEL_COLOR, OUTLINE))

    # 進度
    texts.append((f"已收集：{existing} / {SAMPLES_PER_CLASS}", (10, 60), font_medium, COUNT_COLOR, OUTLINE))

    # 狀態
    state_info = {
        "waiting":   (f"等待手部出現... ({hold_count}/{HOLD_FRAMES})", (150, 150, 150)),
        "countdown": ("準備...比出手勢！",                            (0, 200, 255)),
        "recording": (f"錄製中  {len(sequence)}/{SEQUENCE_LENGTH} 幀",  (0, 220, 80)),
        "cooldown":  (f"冷卻中  {cooldown_count}  ← 換個距離/角度再比一次", (0, 160, 255)),
        "done":      ("此詞彙已完成！按 N 換下一個",                     (0, 220, 80)),
        "paused":    ("暫停偵測，按 S 繼續",                         (220, 180, 0)),
    }
    msg, color = state_info["paused"] if paused else state_info[state]
    texts.append((msg, (10, 95), font_medium, color))

    # 倒數大數字 3-2-1（置中）
    if state == "countdown":
        num = countdown_count // (COUNTDOWN_FRAMES // 3) + 1
        cv2.putText(frame, str(num), (w // 2 - 30, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 200, 255), 8)

    # 有無偵測到手／臉
    hand_text  = f"偵測到 {shown_n} 隻手" if shown_n else "未偵測到手"
    hand_text += "  ｜臉：OK  位置：OK" if has_face else "  ｜臉：無  位置：無"
    hand_color = (100, 220, 255) if shown_n else (80, 80, 200)
    texts.append((hand_text, (10, 130), font_small, hand_color))

    # 實際幀率：太低代表兩幀之間手移動太遠，快速手勢容易跟丟（光線暗時鏡頭也會自動降幀率）
    fps_low = 0 < fps < LOW_FPS
    cv2.putText(frame, f"FPS {fps:.0f}" + ("  (low: add light)" if fps_low else ""), (w - 330, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 255) if fps_low else (160, 160, 160), 2)

    # 臉沒入鏡提醒：位置以臉為基準，手在畫面但臉不在 → 這段錄到的沒有位置資訊
    if shown_n and not has_face and not paused:
        texts.append(("注意：臉沒入鏡，請讓臉也在畫面內（位置要以臉為基準）",
                      (10, h - 180), font_small, (255, 70, 70)))
    elif sample_warn and not paused:
        texts.append((f"注意：{sample_warn}", (10, h - 180), font_small, (255, 170, 0)))

    # 手語提示
    hint_text = get_hint(label)
    if hint_text:
        texts.append((f"提示：{hint_text}", (10, 180), font_large, (0, 0, 220)))

    # 操作提示
    texts.append(("N=下一個 M=上一個 J=跳到指定詞 D=刪除上一筆 S=暫停 Q=離開",
                  (10, h - 90), font_small, KEYS_COLOR, OUTLINE))

    frame = put_texts(frame, texts)
    cv2.imshow("手語資料收集", frame)
    key = read_key()

    if key == ord('q'):
        break
    elif key == ord('n'):
        current_gesture = (current_gesture + 1) % len(GESTURES)
        state = "waiting"
        sequence = []
        hold_count = 0
        paused = True  
        sample_warn = ""
        print(f"[切換] -> {GESTURES[current_gesture]}")
    elif key == ord('m'):
        current_gesture = (current_gesture - 1) % len(GESTURES)
        state = "waiting"
        sequence = []
        hold_count = 0
        paused = True  
        sample_warn = ""
        print(f"[切換] -> {GESTURES[current_gesture]}")
    elif key == ord('d'):
        folder = os.path.join(DATA_DIR, label)
        samples = [f for f in os.listdir(folder) if f.endswith('.npy')]
        if samples:
            # 依檔案修改時間排序（檔名含 session 前綴，不能用數字解析）
            samples.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
            last_file = samples[-1]
            os.remove(os.path.join(folder, last_file))
            sample_warn = ""
            state = "waiting"
            sequence = []
            hold_count = 0
            print(f"[刪除] 已移除 {label} 的上一筆資料：{last_file}")
        else:
            print(f"[刪除] {label} 尚無資料可刪除")
    elif key == ord('j'):
        # 跳到指定詞彙：畫面暫停，回終端機輸入詞名（200 個詞用 N 翻太慢）
        paused = True
        print("\n" + "─" * 50)
        print("跳到指定詞彙。可輸入「詞名」或「編號」，直接按 Enter 取消。")
        未收滿 = [(i, g, _count(g)) for i, g in enumerate(GESTURES) if _count(g) < SAMPLES_PER_CLASS]
        print(f"還沒收滿的詞（前 20 個，共 {len(未收滿)} 個）：")
        for i, g, c in 未收滿[:20]:
            print(f"  {i+1:>3}. {g}（{c} 筆）")
        try:
            ans = input("要跳到：").strip()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        if ans:
            target = None
            if ans.isdigit() and 1 <= int(ans) <= len(GESTURES):
                target = int(ans) - 1
            elif ans in GESTURES:
                target = GESTURES.index(ans)
            if target is None:
                print(f"找不到「{ans}」，維持在目前的詞")
            else:
                current_gesture = target
                state, sequence, hold_count = "waiting", [], 0
                sample_warn = ""
                print(f"[切換] -> {GESTURES[current_gesture]}")
        print("─" * 50 + "\n按 S 繼續收集\n")
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