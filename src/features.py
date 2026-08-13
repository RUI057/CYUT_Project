"""共用特徵處理：抽取、尺度正規化、資料增強。

特徵組成（共 162 維）：
- 前 126 維：雙手 [左手 63][右手 63]，每隻手相對手腕（與舊資料相容）。
- 後  36 維：臉部 18 個非手動標記點（眉/眼/嘴）的 (x,y)，相對鼻尖。

設計原則：
- 儲存到 .npy 的永遠是「raw 特徵」（手減手腕、臉減鼻尖，只做平移）。
  舊資料只有 126 維，載入時用 pad_features 於臉部補 0，維持相容。
- 尺度正規化(normalize_sequence) 只在「餵進模型前」套用
  （訓練載入 + 即時推理），collect/train/test 三邊一致。
"""
import numpy as np

HAND_DIM  = 63              # 21 landmarks * 3 (x,y,z)
HANDS_DIM = HAND_DIM * 2    # 左手 + 右手 = 126

# ── 臉部：非手動標記（Non-Manual Markers）相關的 FaceMesh 關鍵點 ──
# 只取眉/眼/嘴，避免整張 468 點淹沒手部特徵。用 (x,y)，z 對表情較無資訊且雜訊大。
FACE_ORIGIN = 1            # 鼻尖，作為平移原點
FACE_POINTS = [
    33, 133, 159, 145,    # 左眼：外角、內角、上眼皮、下眼皮
    263, 362, 386, 374,   # 右眼：外角、內角、上眼皮、下眼皮
    105, 107,             # 左眉（眉峰、眉頭）
    334, 336,             # 右眉
    61, 291,              # 嘴角（左、右）
    0, 17,                # 上唇外緣、下唇外緣
    13, 14,               # 上唇內緣、下唇內緣
]
FACE_EYE_L, FACE_EYE_R = 0, 4     # 兩眼外角在 FACE_POINTS 內的索引（供尺度用）
FACE_DIM = len(FACE_POINTS) * 2   # 18 * 2 = 36

FEAT_DIM = HANDS_DIM + FACE_DIM    # 126 + 36 = 162


def extract_two_hands(results):
    """從 MediaPipe Hands 結果抽出 raw 手部特徵 (126,)：每隻手相對手腕。

    左手放前 63 維、右手放後 63 維；沒偵測到的手填 0。
    """
    left  = [0.0] * HAND_DIM
    right = [0.0] * HAND_DIM
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


def extract_face(results):
    """從 MediaPipe FaceMesh 結果抽出 raw 臉部特徵 (36,)：18 點相對鼻尖的 (x,y)。

    沒偵測到臉就回傳全 0。results 為 face_mesh.process() 的輸出。
    """
    out = [0.0] * FACE_DIM
    lms = getattr(results, "multi_face_landmarks", None)
    if lms:
        face = lms[0].landmark
        ox, oy = face[FACE_ORIGIN].x, face[FACE_ORIGIN].y
        vals = []
        for idx in FACE_POINTS:
            vals += [face[idx].x - ox, face[idx].y - oy]
        out = vals
    return out


def extract_features(hands_results, face_results):
    """組合雙手 + 臉部 → 162 維 raw 特徵。"""
    return extract_two_hands(hands_results) + extract_face(face_results)


def pad_features(seq):
    """把舊的 126 維序列在臉部補 0，補到 FEAT_DIM(162)。已是 162 則原樣回傳。"""
    seq = np.asarray(seq, dtype=np.float32)
    if seq.shape[-1] >= FEAT_DIM:
        return seq
    pad = np.zeros((seq.shape[0], FEAT_DIM - seq.shape[1]), dtype=np.float32)
    return np.concatenate([seq, pad], axis=1)


def normalize_sequence(seq):
    """尺度正規化：手部除以手大小、臉部除以雙眼外角距離，消除離鏡頭遠近影響。

    seq: (T, 162) raw 特徵（若傳入 126 維會先補 0）→ 回傳 (T, 162)。
    沒偵測到的手/臉（全 0）維持 0。
    """
    seq = pad_features(seq)
    out = seq.copy()
    T = len(seq)

    # 雙手：每隻手除以「各 landmark 到手腕的平面距離平均」
    for off in (0, HAND_DIM):
        hand    = out[:, off:off + HAND_DIM].reshape(T, 21, 3)
        present = np.abs(hand).reshape(T, -1).sum(1) > 1e-6
        scale = np.sqrt(hand[:, :, 0] ** 2 + hand[:, :, 1] ** 2).mean(1)  # (T,)
        safe  = np.where(scale > 1e-6, scale, 1.0)
        normed = (hand / safe[:, None, None]).reshape(T, HAND_DIM)
        out[:, off:off + HAND_DIM] = np.where(present[:, None], normed,
                                              out[:, off:off + HAND_DIM])

    # 臉部：除以雙眼外角距離（inter-ocular distance），對頭部大小/遠近不變
    face    = out[:, HANDS_DIM:].reshape(T, len(FACE_POINTS), 2)
    present = np.abs(face).reshape(T, -1).sum(1) > 1e-6
    iod   = np.sqrt(((face[:, FACE_EYE_L] - face[:, FACE_EYE_R]) ** 2).sum(1))  # (T,)
    safe  = np.where(iod > 1e-6, iod, 1.0)
    normed = (face / safe[:, None, None]).reshape(T, FACE_DIM)
    out[:, HANDS_DIM:] = np.where(present[:, None], normed, out[:, HANDS_DIM:])

    return out


def _time_warp(seq, rng):
    """時間軸隨機伸縮，模擬手勢速度快慢。"""
    T = len(seq)
    speed   = rng.uniform(0.85, 1.15)
    new_idx = np.clip(np.linspace(0, T - 1, T) / speed, 0, T - 1)
    base    = np.arange(T)
    return np.stack(
        [np.interp(new_idx, base, seq[:, c]) for c in range(seq.shape[1])],
        axis=1
    ).astype(np.float32)


def augment_sequence(seq, rng):
    """訓練用資料增強：時間伸縮 + 平面旋轉 + 微抖動 + 隨機掉手/掉臉。

    所有增強都在 raw 特徵上做，之後再 normalize_sequence。
    輸入若為 126 維（舊資料）會先補 0 到 162。
    """
    seq = _time_warp(pad_features(seq), rng)
    T   = len(seq)

    # 平面旋轉（模擬攝影機/頭部些微傾斜），手與臉同角度
    ang  = rng.uniform(-0.20, 0.20)            # 約 ±11 度
    c, s = np.cos(ang), np.sin(ang)
    hands = seq[:, :HANDS_DIM].reshape(T, 2, 21, 3)
    hx = hands[..., 0] * c - hands[..., 1] * s
    hy = hands[..., 0] * s + hands[..., 1] * c
    hands[..., 0], hands[..., 1] = hx, hy
    seq[:, :HANDS_DIM] = hands.reshape(T, HANDS_DIM)
    face = seq[:, HANDS_DIM:].reshape(T, len(FACE_POINTS), 2)
    fx = face[..., 0] * c - face[..., 1] * s
    fy = face[..., 0] * s + face[..., 1] * c
    face[..., 0], face[..., 1] = fx, fy
    seq[:, HANDS_DIM:] = face.reshape(T, FACE_DIM)

    # 高斯抖動，只加在有偵測到的部分（避免把空手/空臉攪成雜訊）
    noise = rng.normal(0, 0.008, seq.shape).astype(np.float32)
    for off in (0, HAND_DIM):
        present = (np.abs(seq[:, off:off + HAND_DIM]).sum(1, keepdims=True) > 1e-6)
        seq[:, off:off + HAND_DIM] += noise[:, off:off + HAND_DIM] * present
    fpresent = (np.abs(seq[:, HANDS_DIM:]).sum(1, keepdims=True) > 1e-6)
    seq[:, HANDS_DIM:] += noise[:, HANDS_DIM:] * fpresent

    # 隨機掉手：把某一隻手在一段連續幀清零，模擬即時偵測掉手
    if rng.random() < 0.20:
        off  = int(rng.integers(0, 2)) * HAND_DIM
        span = int(T * rng.uniform(0.2, 0.6))
        start = int(rng.integers(0, max(1, T - span + 1)))
        seq[start:start + span, off:off + HAND_DIM] = 0.0

    # 隨機掉臉：整段清零，模擬臉沒入鏡/偵測失敗，也讓模型相容臉部補 0 的舊資料
    if rng.random() < 0.25:
        seq[:, HANDS_DIM:] = 0.0

    return seq.astype(np.float32)
