"""共用特徵處理：抽取、尺度正規化、資料增強。

設計原則：
- 儲存到 .npy 的永遠是「raw 特徵」（每隻手減手腕，平移正規化），
  這樣舊資料與新資料相容。
- 尺度正規化(normalize_sequence) 只在「餵進模型前」套用
  （訓練載入 + 即時推理），collect/train/test 三邊一致。
"""
import numpy as np

HAND_DIM = 63          # 21 landmarks * 3 (x,y,z)
FEAT_DIM = HAND_DIM * 2  # 左手 + 右手


def extract_two_hands(results):
    """從 MediaPipe 結果抽出 raw 特徵 (126,)：每隻手相對手腕。

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


def normalize_sequence(seq):
    """尺度正規化：每隻手除以自身大小，消除離鏡頭遠近的影響。

    seq: (T, 126) raw 特徵 → 回傳 (T, 126) 正規化後特徵。
    沒偵測到的手（全 0）維持 0。
    """
    seq = np.asarray(seq, dtype=np.float32)
    out = seq.copy()
    T = len(seq)
    for off in (0, HAND_DIM):
        hand    = out[:, off:off + HAND_DIM].reshape(T, 21, 3)
        present = np.abs(hand).reshape(T, -1).sum(1) > 1e-6
        # 手大小 = 各 landmark 到手腕(原點)的平面距離平均
        scale = np.sqrt(hand[:, :, 0] ** 2 + hand[:, :, 1] ** 2).mean(1)  # (T,)
        safe  = np.where(scale > 1e-6, scale, 1.0)
        normed = (hand / safe[:, None, None]).reshape(T, HAND_DIM)
        out[:, off:off + HAND_DIM] = np.where(present[:, None], normed,
                                              out[:, off:off + HAND_DIM])
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
    """訓練用資料增強：時間伸縮 + 平面旋轉 + 微抖動（僅作用在有手的幀）。

    所有增強都在 raw 特徵上做，之後再 normalize_sequence。
    """
    seq = _time_warp(np.asarray(seq, dtype=np.float32), rng)
    T   = len(seq)

    # 平面旋轉（模擬攝影機/手腕些微傾斜），左右手同角度
    ang  = rng.uniform(-0.20, 0.20)            # 約 ±11 度
    c, s = np.cos(ang), np.sin(ang)
    g = seq.reshape(T, 2, 21, 3)               # (T, 兩隻手, 21, xyz)
    x = g[..., 0] * c - g[..., 1] * s
    y = g[..., 0] * s + g[..., 1] * c
    g[..., 0], g[..., 1] = x, y
    seq = g.reshape(T, FEAT_DIM)

    # 高斯抖動，只加在有偵測到手的部分（避免把空手攪成雜訊）
    noise = rng.normal(0, 0.008, seq.shape).astype(np.float32)
    for off in (0, HAND_DIM):
        present = (np.abs(seq[:, off:off + HAND_DIM]).sum(1, keepdims=True) > 1e-6)
        seq[:, off:off + HAND_DIM] += noise[:, off:off + HAND_DIM] * present

    return seq.astype(np.float32)
