import streamlit as st
import cv2
import json
import time
import os
from datetime import datetime
from src.camera import has_hand_from_array, recognize_local_from_array, reset_buffer
from src.claude_api import polish_sentence
from src.tts import speak

st.set_page_config(
    page_title="台灣手語即時翻譯",
    page_icon="🤟",
    layout="wide"
)

st.title("🤟 台灣手語即時翻譯系統")
st.caption("本地模型即時辨識 + Gemini AI 語意修正")

# ── Session state 初始化 ──────────────────────
for key, default in {
    "word_buffer":       [],
    "history":           [],
    "current_sentence":  "",
    "status":            "待機中",
    "running":           False,
    "last_word":         "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── 版面 ──────────────────────────────────────
col_cam, col_right = st.columns([3, 2])

with col_cam:
    st.subheader("📷 即時畫面")
    frame_placeholder  = st.empty()
    status_placeholder = st.empty()

with col_right:
    st.subheader("🔤 辨識詞彙")
    words_placeholder = st.empty()

    st.subheader("💬 翻譯結果")
    result_placeholder = st.empty()

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        start_btn = st.button("▶ 開始", use_container_width=True, type="primary")
    with col_b2:
        stop_btn = st.button("⏹ 停止", use_container_width=True)

    manual_translate = st.button("✅ 立即翻譯", use_container_width=True)
    clear_btn        = st.button("🗑️ 清除",    use_container_width=True)

st.divider()
st.subheader("📋 翻譯紀錄")
history_placeholder = st.empty()

# ── 控制按鈕 ──────────────────────────────────
if start_btn:
    st.session_state.running = True
    reset_buffer()
if stop_btn:
    st.session_state.running = False
if clear_btn:
    st.session_state.word_buffer    = []
    st.session_state.current_sentence = ""
    reset_buffer()
    st.rerun()

def do_translate():
    """翻譯目前 buffer 並存紀錄"""
    if not st.session_state.word_buffer:
        return
    sentence = polish_sentence(st.session_state.word_buffer)
    st.session_state.current_sentence = sentence
    record = {
        "time":     datetime.now().strftime("%H:%M:%S"),
        "words":    st.session_state.word_buffer.copy(),
        "sentence": sentence
    }
    st.session_state.history.append(record)
    os.makedirs("history", exist_ok=True)
    with open("history/records.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    speak(sentence)
    st.session_state.word_buffer = []
    reset_buffer()

if manual_translate:
    with st.spinner("翻譯中..."):
        do_translate()

# ── 即時辨識主迴圈 ────────────────────────────
RECOG_INTERVAL   = 0.1   # 秒：多久送一幀給本地模型（約 10 fps）
AUTO_TRANSLATE_GAP = 3.0 # 秒：手消失幾秒後自動翻譯

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    last_recog_time = 0
    last_hand_time  = time.time()
    auto_translated = False

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            status_placeholder.error("攝影機讀取失敗")
            break

        frame     = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        now       = time.time()

        # 偵測手
        hand_found = has_hand_from_array(frame_rgb)

        # 畫面狀態提示
        display = frame_rgb.copy()
        if hand_found:
            last_hand_time  = now
            auto_translated = False
            cv2.putText(display, "HAND DETECTED", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 80), 2)
        else:
            cv2.putText(display, "No hand", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 80, 80), 2)

        # 目前 buffer 詞彙疊在畫面底部
        if st.session_state.word_buffer:
            label = " > ".join(st.session_state.word_buffer[-5:])
            cv2.putText(display, label, (10, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2)

        frame_placeholder.image(display, channels="RGB", use_container_width=True)

        # 本地模型辨識（有手 + 間隔夠了才跑）
        if hand_found and (now - last_recog_time > RECOG_INTERVAL):
            word = recognize_local_from_array(frame_rgb)
            if word and word != "？":
                if word != st.session_state.last_word:
                    st.session_state.word_buffer.append(word)
                    st.session_state.last_word = word
                    status_placeholder.success(f"辨識到：**{word}**")
            last_recog_time = now

        # 自動翻譯：手消失超過 AUTO_TRANSLATE_GAP 秒且 buffer 有詞
        gap = now - last_hand_time
        if (not hand_found and gap > AUTO_TRANSLATE_GAP
                and st.session_state.word_buffer and not auto_translated):
            status_placeholder.warning("自動翻譯中...")
            do_translate()
            auto_translated = True
            status_placeholder.success("✅ 翻譯完成")

        # 更新右側面板
        words_placeholder.info(
            "目前詞彙：" + " → ".join(st.session_state.word_buffer)
            if st.session_state.word_buffer else "等待手語輸入..."
        )
        if st.session_state.current_sentence:
            result_placeholder.success(f"**{st.session_state.current_sentence}**")

        # 更新紀錄表格
        if st.session_state.history:
            rows = ""
            for r in reversed(st.session_state.history[-10:]):
                rows += f"| {r['time']} | {' '.join(r['words'])} | {r['sentence']} |\n"
            history_placeholder.markdown(
                "| 時間 | 詞彙 | 翻譯 |\n|------|------|------|\n" + rows
            )

    cap.release()

else:
    status_placeholder.info("按「▶ 開始」啟動即時辨識")
    words_placeholder.info(
        "目前詞彙：" + " → ".join(st.session_state.word_buffer)
        if st.session_state.word_buffer else "等待手語輸入..."
    )
    if st.session_state.current_sentence:
        result_placeholder.success(f"**{st.session_state.current_sentence}**")
    if st.session_state.history:
        rows = ""
        for r in reversed(st.session_state.history[-10:]):
            rows += f"| {r['time']} | {' '.join(r['words'])} | {r['sentence']} |\n"
        history_placeholder.markdown(
            "| 時間 | 詞彙 | 翻譯 |\n|------|------|------|\n" + rows
        )
