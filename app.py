"""台灣手語即時辨識 → 語義翻譯網頁。

流程：攝影機 → LSTM 辨識手語詞 → 累積詞彙序列 →
      本機語言模型轉成自然中文句子（可切換 Gemini，見 src/translate.py）→ gTTS 朗讀。

執行：streamlit run app.py
"""
import os
import sys
import json
from datetime import datetime

import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)   # 翻譯紀錄 history/ 等相對路徑以專案資料夾為準，不管從哪裡啟動

import streamlit as st
from src import camera
from src import translate
from src.tts import synthesize
from src.video import open_camera, camera_help

RECORDS_PATH = "history/records.json"

st.set_page_config(page_title="手語即時翻譯", layout="wide")


# ── 歷史紀錄 ──────────────────────────────────
def load_history():
    if not os.path.exists(RECORDS_PATH):
        return []
    rows = []
    with open(RECORDS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return list(reversed(rows)) 


def append_history(record):
    os.makedirs("history", exist_ok=True)
    with open(RECORDS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── 初始化 session 狀態 ───────────────────────
def init_state():
    ss = st.session_state
    ss.setdefault("words", [])
    ss.setdefault("sentence", "")
    ss.setdefault("sentence_by", "")
    ss.setdefault("audio_bytes", None)
    ss.setdefault("play_audio", False)
    ss.setdefault("history", load_history())
    ss.setdefault("cap", None)
    if "recognizer" not in ss:
        ss.recognizer = camera.SignRecognizer()


init_state()
ss = st.session_state


@st.cache_resource
def _warm_translator():
    """整個程式只做一次：背景載入本機語言模型，第一次翻譯不用等。"""
    translate.warmup()
    return True


_warm_translator()


# ── 翻譯：詞彙序列 → 句子 → 語音 ───────────────
def translate_words():
    words = list(ss.words)
    if not words:
        return
    sentence, backend = translate.polish(words)
    ss.sentence = sentence
    ss.sentence_by = backend
    ss.audio_bytes = synthesize(sentence)
    ss.play_audio = True
    record = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "words": words,
        "sentence": sentence,
        "backend": backend,
    }
    append_history(record)
    ss.history.insert(0, record)
    ss.words = []
    ss.recognizer.reset()


# ── 側邊欄設定 ────────────────────────────────
st.sidebar.title("設定")
rec = ss.recognizer
rec.conf_thresh    = st.sidebar.slider("信心門檻", 0.3, 1.0, float(round(rec.conf_thresh, 2)), 0.01,
                                        help="越高越保守，誤判少但較難觸發")
rec.confirm_frames = st.sidebar.slider("連續確認幀數", 5, 30, rec.confirm_frames, 1,
                                        help="同一個詞連續幾幀才算數，越高越穩")
idle_to_translate  = st.sidebar.slider("自動翻譯停頓（幀）", 10, 60, 25, 5,
                                        help="手離開畫面這麼多幀後，自動把累積詞彙翻成句子")
st.sidebar.divider()
st.sidebar.caption("語義修正：" + translate.backend_status())
st.sidebar.caption("詞彙庫（" + str(len(camera.get_classes())) + " 個）")
st.sidebar.write("、".join(camera.get_classes()) or "（模型未載入）")


# ── 標題 ──────────────────────────────────────
st.title("手語翻譯")
st.caption("MediaPipe + LSTM 辨識手語詞彙，本機語言模型語義修正成自然中文，gTTS 朗讀。")

if not camera.is_ready():
    st.error("找不到模型 data/model.pkl，請先執行 `python train_model.py`。")
    st.stop()

left, right = st.columns([3, 2], gap="large")

frame_ph   = left.empty()
caption_ph = left.empty()

right.subheader("辨識到的詞彙")
words_ph    = right.empty()
btn_cols    = right.columns(3)
right.subheader("翻譯句子")
sentence_ph = right.empty()
audio_ph    = right.empty()


# ── 控制按鈕（在攝影機迴圈之前處理，確保執行時也能點擊）──
if btn_cols[0].button("翻譯", width="stretch", disabled=not ss.words):
    translate_words()
if btn_cols[1].button("朗讀", width="stretch", disabled=not ss.sentence):
    ss.audio_bytes = synthesize(ss.sentence)
    ss.play_audio = True
if btn_cols[2].button("清除", width="stretch"):
    ss.words = []
    ss.sentence = ""
    ss.sentence_by = ""
    ss.audio_bytes = None
    ss.recognizer.reset()


# ── 渲染目前詞彙 / 句子 / 音訊 ──────────────────
def render_words():
    if ss.words:
        words_ph.markdown(
            " ".join(f"<span style='background:#1f6feb;color:#fff;padding:4px 12px;"
                     f"border-radius:14px;margin:3px;display:inline-block;font-size:18px'>{w}</span>"
                     for w in ss.words),
            unsafe_allow_html=True,
        )
    else:
        words_ph.markdown("_尚未辨識到詞彙，請對著鏡頭比手語_")


render_words()


def sentence_md():
    if not ss.sentence:
        return ""
    return f"### {ss.sentence}\n\n:gray[由{ss.sentence_by}產生]" if ss.sentence_by else f"### {ss.sentence}"


sentence_ph.markdown(sentence_md() or "_（翻譯結果會顯示在這裡）_")

# 音訊：只有剛翻譯/剛朗讀時自動播放一次
if ss.audio_bytes:
    audio_ph.audio(ss.audio_bytes, format="audio/mp3", autoplay=ss.play_audio)
ss.play_audio = False


# ── 攝影機開關 ────────────────────────────────
run = right.checkbox("啟動攝影機", value=False)

if run and ss.cap is None:
    ss.cap = open_camera()          # Windows 優先 DirectShow，見 src/video.py
    if ss.cap is None:
        frame_ph.error(camera_help())
    ss.recognizer.reset()
if not run and ss.cap is not None:
    ss.cap.release()
    ss.cap = None
    frame_ph.info("攝影機已關閉。勾選「啟動攝影機」開始辨識。")

if not run and ss.cap is None and not ss.sentence:
    frame_ph.info("勾選右側「啟動攝影機」開始辨識。比完一段手語後，手離開畫面即自動翻譯。")

# ── 歷史紀錄 ──────────────────────────────────
with st.expander(f"翻譯紀錄（{len(ss.history)} 筆）", expanded=False):
    if ss.history:
        for r in ss.history[:20]:
            st.markdown(f"`{r.get('time','')}`　{' '.join(r.get('words', []))}　→　**{r.get('sentence','')}**")
    else:
        st.caption("尚無紀錄")

# ── 即時辨識迴圈（單一連續迴圈，只更新 placeholder，不重跑整頁 → 不閃爍）──
# 取消勾選或點按鈕時，下一次 .image() 會被 Streamlit 中斷並自動重跑腳本，
# 因此迴圈內不需（也不可）呼叫 st.rerun()。
if run and ss.cap is not None:
    cap = ss.cap
    while True:
        ok, frame = cap.read()
        if not ok:
            frame_ph.warning("讀取攝影機失敗")
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        state = rec.process(rgb)

        camera.draw_landmarks(frame, state["results"], state.get("face_results"))

        # 確認到新詞 → 加入序列（避免連續重複）
        if state["confirmed"]:
            if not ss.words or ss.words[-1] != state["confirmed"]:
                ss.words.append(state["confirmed"])
                render_words()

        # 手離開夠久且有累積詞彙 → 自動翻譯，並就地更新句子/語音
        if state["idle_frames"] >= idle_to_translate and ss.words:
            translate_words()
            render_words()
            sentence_ph.markdown(sentence_md())
            if ss.audio_bytes:
                audio_ph.audio(ss.audio_bytes, format="audio/mp3", autoplay=True)
            ss.play_audio = False   # 已就地播放，避免之後重跑時重播

        # 畫面顯示
        frame_ph.image(frame, channels="BGR", width="stretch")
        status = (f"偵測到 {state['hand_n']} 隻手　"
                  if state["has_hand"] else "未偵測到手　")
        if state["cooldown"] > 0:
            status += f"冷卻 {state['cooldown']}　"
        elif state["confirm_word"]:
            status += f"確認中：**{state['confirm_word']}** {state['confirm_pct']}%　"
        status += f"Buffer {state['buffer_pct']}%　Motion {state['motion']:.4f}"
        caption_ph.markdown(status)