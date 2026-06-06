import mediapipe as mp
import numpy as np
import pickle

mp_hands = mp.solutions.hands
_hands_static  = mp_hands.Hands(static_image_mode=True,  max_num_hands=2, min_detection_confidence=0.6)
_hands_dynamic = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# 載入本地模型
try:
    with open("data/model.pkl", "rb") as f:
        _payload  = pickle.load(f)
    _model    = _payload["model"]
    _labels   = _payload["labels"]
    _seq_len  = _payload["seq_len"]
    print(f"[模型] 載入成功，詞彙：{_labels}")
except FileNotFoundError:
    _model = _labels = _seq_len = None
    print("[警告] 找不到模型，請先執行 train_model.py")

# 滑動窗口緩衝區
_sequence_buffer = []

# ── 共用：萃取雙手特徵 ──────────────────────────
def _extract_two_hands(results):
    left  = [0.0] * 63
    right = [0.0] * 63
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handedness in zip(results.multi_hand_landmarks,
                                        results.multi_handedness):
            label = handedness.classification[0].label
            wrist = hand_lms.landmark[0]
            features = []
            for lm in hand_lms.landmark:
                features.append(lm.x - wrist.x)
                features.append(lm.y - wrist.y)
                features.append(lm.z - wrist.z)
            if label == "Left":
                left = features
            else:
                right = features
    return left + right  # 126 維

# ── 接收 numpy array（給 app.py OpenCV 模式用）──
def has_hand_from_array(frame_rgb: np.ndarray) -> bool:
    """判斷 numpy RGB 畫面中是否有手"""
    result = _hands_static.process(frame_rgb)
    return result.multi_hand_landmarks is not None

def recognize_local_from_array(frame_rgb: np.ndarray) -> str:
    """
    即時辨識 numpy RGB 畫面（滑動窗口，不呼叫任何 API）。
    回傳辨識詞彙；尚未累積足夠幀數或信心不足時回傳空字串。
    """
    global _sequence_buffer

    if _model is None:
        return "？"

    try:
        result = _hands_dynamic.process(frame_rgb)
        frame_data = _extract_two_hands(result)
        _sequence_buffer.append(frame_data)

        if len(_sequence_buffer) > _seq_len:
            _sequence_buffer.pop(0)

        if len(_sequence_buffer) == _seq_len:
            X     = np.array(_sequence_buffer).flatten().reshape(1, -1)
            proba = _model.predict_proba(X).max()
            pred  = _model.predict(X)[0]
            if proba > 0.75:
                return pred

        return ""

    except Exception as e:
        print(f"[辨識錯誤] {e}")
        return ""

def reset_buffer():
    """清除序列緩衝區"""
    global _sequence_buffer
    _sequence_buffer = []
