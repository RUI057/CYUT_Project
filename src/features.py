"""共用特徵處理：抽取、尺度正規化、資料增強。

儲存格式（.npy，每筆 30 幀 × 126 維）：
- 雙手 [左手 63][右手 63]，每隻手 21 個點的 (x,y,z)，相對手腕 → 手形與手的轉動。

歷史：2026-07 ～ 10 曾加過臉部非手動標記（眉/眼/嘴 36 維）與手相對臉的位置（8 維），
存成 162 / 170 維；2026-10 整個拿掉（不再偵測臉）。已錄的舊 .npy 不必改檔：
讀取時 hands_only 只留前 126 維，其餘忽略（舊資料的手部欄位與現在完全相同）。

設計原則：
- 儲存到 .npy 的永遠是「raw 特徵」（手減手腕）。
- 尺度正規化(normalize_sequence) 只在「餵進模型前」套用
  （訓練載入 + 即時推理），collect/train/test 三邊一致。
- 模型輸入由 to_model_input 決定，維度就是 FEAT_DIM。
"""
import numpy as np

HAND_DIM  = 63              # 21 landmarks * 3 (x,y,z)
HANDS_DIM = HAND_DIM * 2    # 左手 + 右手 = 126
FEAT_DIM  = HANDS_DIM       # 現行儲存格式，也是模型輸入維度
LEGACY_DIMS = (162, 170)    # 舊格式（手 + 臉 [+ 位置]）；只讀取，不再產生
VALID_DIMS  = (HANDS_DIM,) + LEGACY_DIMS   # 可接受的 .npy 寬度（新舊格式）


# ── 動作門檻：推理時手幾乎不動就不預測，避免把「手舉著沒在比」猜成某個詞 ──
# 但有些手勢本來就幾乎靜止（誰：拇指貼臉頰），全部詞用同一個門檻會把它們擋掉
# （實測「誰」72% 的真實樣本低於 0.003）。所以每個詞另有門檻：
#   該詞訓練資料第 MOTION_PCT 百分位的動作量，且不超過全域門檻。
# 動態的詞第 5 百分位本來就 > 0.003 → 維持 0.003，行為不變；只有靜態的詞會放寬。
MOTION_THRESH = 0.003
MOTION_PCT    = 5


def hand_motion(arr):
    """手部動作量：raw 雙手 126 維逐維標準差的平均（舊資料多的臉部、位置欄位不算）。"""
    arr = np.asarray(arr, dtype=np.float32)
    return float(np.std(arr[:, :HANDS_DIM], axis=0).mean())


def class_motion_thresholds(motions_by_class, base=MOTION_THRESH, pct=MOTION_PCT):
    """{詞: [各樣本動作量]} → {詞: 門檻}；門檻 = min(base, 第 pct 百分位)。"""
    return {c: min(base, float(np.percentile(m, pct)))
            for c, m in motions_by_class.items() if len(m)}

# ── MediaPipe 參數：收集與推理必須一致，否則偵測行為不同 → 訓練/推理資料分佈不同 ──
# （只是 dict，本模組不 import mediapipe，訓練端仍不需要裝 mediapipe）
MP_HANDS_KWARGS = dict(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)

# ── 手部軌跡濾波：MediaPipe 逐幀偵測會有雜訊 ──
# 實測既有資料：12% 樣本有幽靈手、21% 有短暫掉手、6% 左右手跳槽。
# 調閾值只能此消彼長（提高 → 幽靈少但掉手多），改用時間軸濾波一次處理。
GHOST_MAX = 2   # 某隻手只出現 ≤ 這麼多幀就消失 → 幽靈手（誤偵測臉/背景），移除
GAP_MAX   = 2   # 手在中間消失 ≤ 這麼多幀又出現 → 掉幀，用前後內插補回


def _hand_slots(results):
    """MediaPipe Hands 結果 → [("Left"|"Right", hand_landmarks), ...]。

    收集、推理都用這裡的分配，確保同一隻手落在同一槽。
    """
    if not (results.multi_hand_landmarks and results.multi_handedness):
        return []
    hands = [(hd.classification[0].label, lms)
             for lms, hd in zip(results.multi_hand_landmarks, results.multi_handedness)]

    # 兩隻手被標成同一邊時，原本後者會直接覆蓋前者 → 一隻手的資料憑空消失。
    # 改依手腕水平位置分配：畫面已鏡像翻轉，使用者的左手在畫面左側（x 較小）。
    if len(hands) == 2 and hands[0][0] == hands[1][0]:
        hands.sort(key=lambda h: h[1].landmark[0].x)
        hands = [("Left", hands[0][1]), ("Right", hands[1][1])]
    return hands


def extract_two_hands(results):
    """從 MediaPipe Hands 結果抽出 raw 手部特徵 (126,)：每隻手相對手腕。

    左手放前 63 維、右手放後 63 維；沒偵測到的手填 0。
    """
    left  = [0.0] * HAND_DIM
    right = [0.0] * HAND_DIM
    for lbl, hand_lms in _hand_slots(results):
        wrist = hand_lms.landmark[0]
        feat  = []
        for lm in hand_lms.landmark:
            feat += [lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z]
        if lbl == "Left":
            left = feat
        else:
            right = feat
    return left + right


def hands_only(seq):
    """任何寬度的 raw 序列（126 / 162 / 170）→ 只留雙手 126 維（舊資料的臉部、位置欄位忽略）。

    回傳 float32，可能是原陣列的 view；要原地修改請先 .copy()。
    """
    seq = np.asarray(seq, dtype=np.float32)
    if seq.shape[-1] < HANDS_DIM:
        raise ValueError(f"特徵寬度 {seq.shape[-1]} < {HANDS_DIM}，不是有效的 raw 特徵")
    return seq[..., :HANDS_DIM]


def _runs(mask):
    """把布林序列切成連續區段 → [(值, 起點, 長度), ...]。"""
    out, i, n = [], 0, len(mask)
    while i < n:
        j = i
        while j < n and mask[j] == mask[i]:
            j += 1
        out.append((bool(mask[i]), i, j - i))
        i = j
    return out


def _hand_slice(k):
    """第 k 隻手（0=左、1=右）在 126 維中的欄位。"""
    return slice(k * HAND_DIM, (k + 1) * HAND_DIM)


def longest_hand_gap(seq):
    """左右手各自「出現過之後又消失、之後再出現」的最長連續缺口幀數，取兩手較大者。

    頭尾沒有手不算（手本來就可能還沒進畫面 / 已經離開）。> GAP_MAX 代表
    clean_hand_tracks 補不回來，通常是動作太快或光線不足，MediaPipe 跟丟了手。
    """
    seq = hands_only(seq)
    longest = 0
    for off in (0, HAND_DIM):
        present = np.abs(seq[:, off:off + HAND_DIM]).sum(1) > 1e-6
        rs = _runs(present)
        for k, (v, _st, ln) in enumerate(rs):
            if not v and 0 < k < len(rs) - 1:
                longest = max(longest, ln)
    return longest


def clean_hand_tracks(seq, ghost_max=GHOST_MAX, gap_max=GAP_MAX):
    """清掉手部偵測雜訊，回傳新陣列（不改原資料；輸出一律是 126 維）。

    對左右手各自處理：
      1. 補掉幀：中間缺 ≤ gap_max 幀、前後都有手 → 線性內插補回
      2. 去幽靈：只出現 ≤ ghost_max 幀的片段 → 清零
    左右手跳槽（同一隻手短暫跑到另一槽）會同時被這兩步修好：
    原槽的缺口被補回，另一槽的短暫出現被清掉。

    先補再去：若先去幽靈，斷斷續續的真實手（有 1 有 0 交錯）會被誤刪成大洞補不回來。
    頭尾的缺手不補（手本來就可能還沒進畫面 / 已經離開）。
    """
    seq = hands_only(seq).copy()
    for k in (0, 1):
        block   = seq[:, _hand_slice(k)]                    # view：直接改就是改 seq
        present = np.abs(block).sum(1) > 1e-6

        rs = _runs(present)
        for i_run, (v, st, ln) in enumerate(rs):
            if not v and ln <= gap_max and 0 < i_run < len(rs) - 1:
                a, b = block[st - 1].copy(), block[st + ln].copy()
                for i in range(ln):
                    w = (i + 1) / (ln + 1)
                    block[st + i] = a * (1 - w) + b * w
                present[st:st + ln] = True

        rs = _runs(present)
        if len(rs) > 1:                                      # 整段都有手就不是幽靈
            for v, st, ln in rs:
                if v and ln <= ghost_max:
                    block[st:st + ln] = 0.0
    return seq


def normalize_sequence(seq):
    """尺度正規化：每隻手除以「各點到手腕的平面距離平均」，消除離鏡頭遠近影響。

    seq: (T, 126 / 162 / 170) raw 特徵（舊資料多的臉部、位置欄位會被丟掉）→ 回傳 (T, 126)。
    沒偵測到的手（全 0）維持 0。
    """
    out = hands_only(seq).copy()
    T = len(out)

    for off in (0, HAND_DIM):
        hand    = out[:, off:off + HAND_DIM].reshape(T, 21, 3)
        present = np.abs(hand).reshape(T, -1).sum(1) > 1e-6
        scale = np.sqrt(hand[:, :, 0] ** 2 + hand[:, :, 1] ** 2).mean(1)  # (T,)
        safe  = np.where(scale > 1e-6, scale, 1.0)
        normed = (hand / safe[:, None, None]).reshape(T, HAND_DIM)
        out[:, off:off + HAND_DIM] = np.where(present[:, None], normed,
                                              out[:, off:off + HAND_DIM])
    return out


def to_model_input(seq):
    """raw 特徵 → 模型輸入 (T, FEAT_DIM)。訓練與推理共用這個入口。"""
    return normalize_sequence(seq)


def _time_warp(seq, rng):
    """時間軸隨機伸縮，模擬手勢速度快慢（線性內插）。"""
    T = len(seq)
    speed   = rng.uniform(0.85, 1.15)
    new_idx = np.clip(np.linspace(0, T - 1, T) / speed, 0, T - 1)
    base    = np.arange(T)
    smooth  = np.stack([np.interp(new_idx, base, seq[:, c]) for c in range(HANDS_DIM)], axis=1)
    return smooth.astype(np.float32)


def _geo_jitter(seq, rng):
    """手形比例／深度／旋轉的隨機擾動（兩隻手各自獨立）。

    不同人的手指長短比例、手腕角度、離鏡頭遠近都不同，而 MediaPipe 的 z 又最不可靠。
    2026-10-25 對照實驗（只有新資料、每場不同人的 48 詞，跨場次 5 折驗證，各設定同切分同種子）：
      現行增強 57.4%（兩個種子 57.9 / 56.9）→ 加上這個 62.9%，五折全部都比現行好；
      更寬的速度＋起訖位移 57.1%（無效）、左右手鏡射 53.1%（有害）所以沒採用。
    """
    seq = seq.copy()
    T = len(seq)
    h = seq[:, :HANDS_DIM].reshape(T, 2, 21, 3)        # 欄位不連續，這裡是副本，最後要寫回
    for k in (0, 1):
        h[:, k, :, 0] *= rng.uniform(0.85, 1.15)
        h[:, k, :, 1] *= rng.uniform(0.85, 1.15)
        h[:, k, :, 2] *= rng.uniform(0.6, 1.4)         # z 最不可靠，擾動最大
        a = rng.uniform(-0.25, 0.25)
        c, s = np.cos(a), np.sin(a)
        x, y = h[:, k, :, 0].copy(), h[:, k, :, 1].copy()
        h[:, k, :, 0], h[:, k, :, 1] = x * c - y * s, x * s + y * c
    seq[:, :HANDS_DIM] = h.reshape(T, HANDS_DIM)
    return seq


def augment_sequence(seq, rng):
    """訓練用資料增強：手形擾動 + 時間伸縮 + 平面旋轉 + 微抖動 + 隨機掉手。

    所有增強都在 raw 特徵上做，之後再 normalize_sequence。
    輸入若為 162 / 170 維（舊資料）只取前 126 維。
    """
    seq = _time_warp(_geo_jitter(hands_only(seq), rng), rng)
    T   = len(seq)

    # 平面旋轉（模擬攝影機些微傾斜）
    ang  = rng.uniform(-0.20, 0.20)            # 約 ±11 度
    c, s = np.cos(ang), np.sin(ang)
    hands = seq.reshape(T, 2, 21, 3)
    hx = hands[..., 0] * c - hands[..., 1] * s
    hy = hands[..., 0] * s + hands[..., 1] * c
    hands[..., 0], hands[..., 1] = hx, hy
    seq = hands.reshape(T, HANDS_DIM)

    # 高斯抖動，只加在有偵測到的手（避免把空手攪成雜訊）
    noise = rng.normal(0, 0.008, seq.shape).astype(np.float32)
    for off in (0, HAND_DIM):
        present = (np.abs(seq[:, off:off + HAND_DIM]).sum(1, keepdims=True) > 1e-6)
        seq[:, off:off + HAND_DIM] += noise[:, off:off + HAND_DIM] * present

    # 隨機掉手：把某一隻手在一段連續幀清零，模擬即時偵測掉手
    if rng.random() < 0.20:
        k    = int(rng.integers(0, 2))
        span = int(T * rng.uniform(0.2, 0.6))
        start = int(rng.integers(0, max(1, T - span + 1)))
        seq[start:start + span, _hand_slice(k)] = 0.0

    return seq.astype(np.float32)
