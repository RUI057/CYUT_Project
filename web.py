"""台灣手語即時辨識 → 語義翻譯網頁（FastAPI + MJPEG 串流）。

流程：攝影機 → LSTM 辨識手語詞 → 累積詞彙序列 →
      本機語言模型轉成自然中文句子（可切換 Gemini，見 src/translate.py）→ gTTS 朗讀。

執行：python web.py            （預設 http://127.0.0.1:8000）
      python web.py --port 9000 --host 0.0.0.0

為什麼用 MJPEG 而不是 Streamlit：
  Streamlit 的執行模型是「整頁重跑」，推即時影像得靠一個不會結束的迴圈撐著，
  按鈕、設定都要跟這個迴圈搶執行權。這裡把職責切開：
  - 擷取執行緒：獨佔攝影機與 MediaPipe（兩者都不是執行緒安全的），負責辨識與編碼
  - /stream：把同一份 JPEG 丟給所有連線（multipart/x-mixed-replace）
  - 其他端點：純粹讀寫共享狀態，隨時可用，不受影像迴圈影響
"""
import argparse
import asyncio
import json
import os
import queue
import sys
import threading
import time
from datetime import datetime

# macOS：AVFoundation 的攝影機授權請求只能在主執行緒發出，但這裡是在背景執行緒開
# 攝影機（主執行緒被 uvicorn 佔著）→ OpenCV 會卡在 "can not spin main run loop
# from other thread" 而開不起來。跳過 OpenCV 自己的授權流程，沿用系統既有授權。
# 必須在 import cv2 之前設定才有效。
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)   # 翻譯紀錄 history/ 等相對路徑以專案資料夾為準，不管從哪裡啟動

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from src import camera
from src import translate
from src.font_utils import get_font, put_text
from src.tts import synthesize
from src.video import open_camera, camera_help

RECORDS_PATH = "history/records.json"
JPEG_QUALITY = 80
BOUNDARY = "frame"
HEARTBEAT_SEC = 0.5      # 畫面沒更新時的重送間隔（見 /stream 的說明）
_font = get_font(34)


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


def placeholder_jpeg(text, sub=""):
    """攝影機沒開/打不開時也要有畫面，否則瀏覽器會一直卡在載入中。"""
    img = np.full((480, 640, 3), 24, np.uint8)
    img = put_text(img, text, (40, 200), _font, (230, 230, 230))
    if sub:
        for i, line in enumerate(sub.split("\n")[:5]):
            img = put_text(img, line, (40, 250 + i * 40), get_font(22), (150, 150, 150))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else b""


# ── 共享狀態 ──────────────────────────────────
class Hub:
    """擷取執行緒寫、HTTP 端點讀。所有欄位都在 lock 底下存取。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = placeholder_jpeg("攝影機未啟動", "按「啟動攝影機」開始辨識")
        self.frame_seq = 0          # 每張新影格 +1，/stream 靠它判斷有沒有新畫面
        self.status = {"has_hand": False, "has_face": False, "hand_n": 0,
                       "confirm_word": "", "confirm_pct": 0, "buffer_pct": 0,
                       "motion": 0.0, "cooldown": 0, "fps": 0.0}
        self.words = []
        self.sentence = ""
        self.sentence_by = ""
        self.audio = None
        self.audio_seq = 0          # 變動時前端才重新抓音訊，避免重複播放
        self.translating = False
        self.cam_error = ""
        self.history = load_history()
        self.idle_to_translate = 25

        self.want_cam = threading.Event()
        self.stop = threading.Event()
        self.recognizer = camera.SignRecognizer()
        self.jobs = queue.Queue(maxsize=1)   # 翻譯工作；滿了就丟棄，不堆積

    # ── 給端點用的小工具 ──
    def snapshot(self):
        with self.lock:
            return {
                "camera_on": self.want_cam.is_set(),
                "cam_error": self.cam_error,
                "words": list(self.words),
                "sentence": self.sentence,
                "sentence_by": self.sentence_by,
                "translating": self.translating,
                "audio_seq": self.audio_seq,
                "has_audio": self.audio is not None,
                "status": dict(self.status),
                "conf_thresh": round(float(self.recognizer.conf_thresh), 3),
                "confirm_frames": int(self.recognizer.confirm_frames),
                "idle_to_translate": int(self.idle_to_translate),
                "history": self.history[:20],
                "classes": camera.get_classes(),
            }

    def set_frame(self, jpeg, status):
        with self.lock:
            self.jpeg = jpeg
            self.frame_seq += 1
            self.status.update(status)

    def show_placeholder(self, text, sub=""):
        with self.lock:
            self.jpeg = placeholder_jpeg(text, sub)
            self.frame_seq += 1

    def queue_translate(self):
        """把翻譯交給 worker。回傳是否真的排進去了。"""
        with self.lock:
            if self.translating or not self.words:
                return False
            words = list(self.words)
            self.words = []
            self.translating = True
        self.recognizer.reset()
        try:
            self.jobs.put_nowait(("polish", words))
            return True
        except queue.Full:
            with self.lock:
                self.words = words
                self.translating = False
            return False


HUB = Hub()


# ── 擷取執行緒：攝影機 + MediaPipe + 辨識 ─────────
def capture_loop(hub: Hub):
    cap = None
    last, fps = time.time(), 0.0
    try:
        while not hub.stop.is_set():
            want = hub.want_cam.is_set()

            if want and cap is None:
                cap = open_camera()
                if cap is None:
                    hub.want_cam.clear()
                    with hub.lock:
                        hub.cam_error = camera_help()
                    hub.show_placeholder("打不開攝影機", camera_help())
                    continue
                with hub.lock:
                    hub.cam_error = ""
                hub.recognizer.reset()

            if not want and cap is not None:
                cap.release()
                cap = None
                hub.recognizer.reset()
                hub.show_placeholder("攝影機已關閉", "按「啟動攝影機」重新開始")

            if cap is None:
                time.sleep(0.05)
                continue

            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = None
                hub.want_cam.clear()
                with hub.lock:
                    hub.cam_error = "讀取攝影機失敗（可能被其他程式佔用）"
                hub.show_placeholder("讀取攝影機失敗", "確認沒有其他程式在用攝影機")
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            state = hub.recognizer.process(rgb)
            camera.draw_landmarks(frame, state["results"], state.get("face_results"))

            # 確認到新詞 → 加入序列（連續同詞只算一次）
            if state["confirmed"]:
                with hub.lock:
                    if not hub.words or hub.words[-1] != state["confirmed"]:
                        hub.words.append(state["confirmed"])

            # 手離開夠久且有累積詞彙 → 自動翻譯（交給 worker，影像不停）
            with hub.lock:
                idle_limit = hub.idle_to_translate
                pending = bool(hub.words)
            if pending and state["idle_frames"] >= idle_limit:
                hub.queue_translate()

            now = time.time()
            dt = now - last
            last = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            enc_ok, buf = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if enc_ok:
                hub.set_frame(buf.tobytes(), {
                    "has_hand": state["has_hand"], "has_face": state["has_face"],
                    "hand_n": state["hand_n"], "confirm_word": state["confirm_word"],
                    "confirm_pct": state["confirm_pct"], "buffer_pct": state["buffer_pct"],
                    "motion": round(float(state["motion"]), 4),
                    "cooldown": state["cooldown"], "fps": round(fps, 1),
                })
    finally:
        if cap is not None:
            cap.release()


# ── 翻譯執行緒：本機語言模型很慢，不能卡住影像或事件迴圈 ──
def worker_loop(hub: Hub):
    while not hub.stop.is_set():
        try:
            kind, payload = hub.jobs.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            if kind == "polish":
                sentence, backend = translate.polish(payload)
                audio = synthesize(sentence) if sentence else None
                record = {"time": datetime.now().strftime("%H:%M:%S"),
                          "words": payload, "sentence": sentence, "backend": backend}
                if sentence:
                    append_history(record)
                with hub.lock:
                    hub.sentence, hub.sentence_by = sentence, backend
                    hub.audio = audio
                    hub.audio_seq += 1
                    if sentence:
                        hub.history.insert(0, record)
            elif kind == "speak":
                audio = synthesize(payload) if payload else None
                with hub.lock:
                    hub.audio = audio
                    hub.audio_seq += 1
        except Exception as e:                      # 翻譯/TTS 失敗不該讓整個服務掛掉
            with hub.lock:
                hub.sentence_by = f"失敗：{type(e).__name__}"
        finally:
            with hub.lock:
                hub.translating = False
            hub.jobs.task_done()


# ── FastAPI ───────────────────────────────────
app = FastAPI(title="手語即時翻譯")
_threads = []


@app.on_event("startup")
def _startup():
    for fn in (capture_loop, worker_loop):
        t = threading.Thread(target=fn, args=(HUB,), daemon=True)
        t.start()
        _threads.append(t)
    threading.Thread(target=translate.warmup, daemon=True).start()  # 預熱，第一次翻譯不用等


@app.on_event("shutdown")
def _shutdown():
    HUB.stop.set()
    HUB.want_cam.clear()
    for t in _threads:
        t.join(timeout=2)


@app.get("/stream")
async def stream():
    """MJPEG：multipart/x-mixed-replace，瀏覽器用 <img src="/stream"> 就能播。"""
    async def gen():
        last_seq, last_sent = -1, 0.0
        while not HUB.stop.is_set():
            with HUB.lock:
                seq, jpeg = HUB.frame_seq, HUB.jpeg
            now = time.monotonic()
            # 沒有新畫面也要定期重送：瀏覽器要收到「下一個 boundary」才會把上一張畫出來，
            # 攝影機關閉時只送一張的話，佔位圖會一直卡著不顯示。
            if seq != last_seq or now - last_sent > HEARTBEAT_SEC:
                last_seq, last_sent = seq, now
                yield (b"--" + BOUNDARY.encode() + b"\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                       + jpeg + b"\r\n")
            await asyncio.sleep(0.005)   # 讓出事件迴圈；靠 frame_seq 判斷有無新畫面

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate",
                 "Pragma": "no-cache"},
    )


@app.get("/state")
async def get_state():
    return JSONResponse(HUB.snapshot())


@app.get("/audio")
async def get_audio():
    with HUB.lock:
        audio = HUB.audio
    if not audio:
        return Response(status_code=204)
    return Response(audio, media_type="audio/mpeg",
                    headers={"Cache-Control": "no-store"})


class CameraReq(BaseModel):
    on: bool


@app.post("/camera")
async def set_camera(req: CameraReq):
    if req.on:
        with HUB.lock:
            HUB.cam_error = ""
        HUB.want_cam.set()
    else:
        HUB.want_cam.clear()
    return {"ok": True}


@app.post("/translate")
async def do_translate():
    return {"ok": HUB.queue_translate()}


@app.post("/speak")
async def do_speak():
    with HUB.lock:
        sentence = HUB.sentence
        busy = HUB.translating
    if not sentence or busy:
        return {"ok": False}
    with HUB.lock:
        HUB.translating = True
    try:
        HUB.jobs.put_nowait(("speak", sentence))
    except queue.Full:
        with HUB.lock:
            HUB.translating = False
        return {"ok": False}
    return {"ok": True}


@app.post("/clear")
async def do_clear():
    with HUB.lock:
        HUB.words = []
        HUB.sentence = ""
        HUB.sentence_by = ""
        HUB.audio = None
        HUB.audio_seq += 1
    HUB.recognizer.reset()
    return {"ok": True}


class Settings(BaseModel):
    conf_thresh: float | None = None
    confirm_frames: int | None = None
    idle_to_translate: int | None = None


@app.post("/settings")
async def set_settings(s: Settings):
    with HUB.lock:
        if s.conf_thresh is not None:
            HUB.recognizer.conf_thresh = max(0.3, min(1.0, s.conf_thresh))
        if s.confirm_frames is not None:
            HUB.recognizer.confirm_frames = max(5, min(30, s.confirm_frames))
        if s.idle_to_translate is not None:
            HUB.idle_to_translate = max(10, min(60, s.idle_to_translate))
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
async def index():
    ready = camera.is_ready()
    warn = "" if ready else (
        '<p class="warn">找不到模型 <code>data/model.pkl</code>，'
        '請先執行 <code>python train_model.py</code>。畫面可以開，但不會辨識。</p>')
    return PAGE.replace("<!--WARN-->", warn).replace(
        "<!--BACKEND-->", translate.backend_status())


PAGE = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>手語即時翻譯</title>
<style>
  :root { --bg:#0f1116; --panel:#181b22; --line:#2a2f3a; --fg:#e6e8ee;
          --muted:#9aa3b2; --accent:#1f6feb; --ok:#2ea043; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font-family:-apple-system,"Noto Sans TC","Microsoft JhengHei",sans-serif; }
  header { padding:14px 20px; border-bottom:1px solid var(--line); }
  h1 { margin:0; font-size:19px; }
  header p { margin:4px 0 0; color:var(--muted); font-size:13px; }
  .warn { color:#f0883e; }
  .wrap { display:flex; gap:18px; padding:18px; flex-wrap:wrap; align-items:flex-start; }
  .video { flex:1 1 560px; min-width:320px; }
  .video img { width:100%; border-radius:10px; background:#000; display:block; }
  .side { flex:1 1 340px; min-width:300px; display:flex; flex-direction:column; gap:14px; }
  .panel { background:var(--panel); border:1px solid var(--line);
           border-radius:10px; padding:14px; }
  .panel h2 { margin:0 0 10px; font-size:14px; color:var(--muted); font-weight:600; }
  .chips { display:flex; flex-wrap:wrap; gap:6px; min-height:34px; }
  .chip { background:var(--accent); padding:5px 13px; border-radius:14px; font-size:16px; }
  .sentence { font-size:21px; line-height:1.5; min-height:32px; }
  .by { color:var(--muted); font-size:12px; margin-top:6px; }
  .row { display:flex; gap:8px; }
  button { flex:1; padding:10px; border:1px solid var(--line); border-radius:8px;
           background:#222733; color:var(--fg); font-size:14px; cursor:pointer; }
  button:hover:not(:disabled) { border-color:var(--accent); }
  button:disabled { opacity:.4; cursor:default; }
  button.primary { background:var(--accent); border-color:var(--accent); }
  button.on { background:var(--ok); border-color:var(--ok); }
  .stat { color:var(--muted); font-size:13px; font-variant-numeric:tabular-nums; }
  label.set { display:block; font-size:12px; color:var(--muted); margin:10px 0 2px; }
  input[type=range] { width:100%; }
  .hist { max-height:200px; overflow:auto; font-size:13px; }
  .hist div { padding:4px 0; border-bottom:1px solid var(--line); }
  .hist code { color:var(--muted); }
  details summary { cursor:pointer; color:var(--muted); font-size:13px; }
  .cls { color:var(--muted); font-size:12px; line-height:1.7; max-height:92px; overflow:auto; }
</style>
</head>
<body>
<header>
  <h1>手語翻譯</h1>
  <p>MediaPipe + LSTM 辨識手語詞彙，語義修正：<!--BACKEND-->，gTTS 朗讀。</p>
  <!--WARN-->
</header>

<div class="wrap">
  <div class="video">
    <img id="cam" src="/stream" alt="攝影機畫面">
    <p class="stat" id="stat">—</p>
  </div>

  <div class="side">
    <div class="panel">
      <div class="row">
        <button id="camBtn" class="primary">啟動攝影機</button>
      </div>
      <p class="stat" id="camErr" style="white-space:pre-line"></p>
    </div>

    <div class="panel">
      <h2>辨識到的詞彙</h2>
      <div class="chips" id="words"></div>
      <div class="row" style="margin-top:12px">
        <button id="btnT">翻譯</button>
        <button id="btnS">朗讀</button>
        <button id="btnC">清除</button>
      </div>
    </div>

    <div class="panel">
      <h2>翻譯句子</h2>
      <div class="sentence" id="sent">—</div>
      <div class="by" id="by"></div>
      <audio id="au" style="display:none"></audio>
    </div>

    <div class="panel">
      <h2>設定</h2>
      <label class="set">信心門檻 <span id="vc"></span></label>
      <input type="range" id="sc" min="0.3" max="1" step="0.005">
      <label class="set">連續確認幀數 <span id="vf"></span></label>
      <input type="range" id="sf" min="5" max="30" step="1">
      <label class="set">自動翻譯停頓（幀）<span id="vi"></span></label>
      <input type="range" id="si" min="10" max="60" step="5">
    </div>

    <div class="panel">
      <details>
        <summary>翻譯紀錄 / 詞彙庫</summary>
        <div class="hist" id="hist" style="margin-top:10px"></div>
        <div class="cls" id="cls" style="margin-top:10px"></div>
      </details>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
let lastAudio = -1, dragging = false;

async function post(url, body) {
  await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'},
                    body: JSON.stringify(body || {})});
  tick();
}

$('camBtn').onclick = () => post('/camera', {on: !$('camBtn').classList.contains('on')});
$('btnT').onclick = () => post('/translate');
$('btnS').onclick = () => post('/speak');
$('btnC').onclick = () => post('/clear');

for (const [el, key, fmt] of [['sc','conf_thresh',v=>(+v).toFixed(3)],
                              ['sf','confirm_frames',v=>v],
                              ['si','idle_to_translate',v=>v]]) {
  const input = $(el);
  input.addEventListener('input', () => {
    dragging = true;
    $({sc:'vc',sf:'vf',si:'vi'}[el]).textContent = fmt(input.value);
  });
  input.addEventListener('change', async () => {
    await post('/settings', {[key]: key === 'conf_thresh' ? +input.value : +input.value});
    dragging = false;
  });
}

async function tick() {
  let s;
  try { s = await (await fetch('/state')).json(); } catch { return; }

  $('camBtn').textContent = s.camera_on ? '關閉攝影機' : '啟動攝影機';
  $('camBtn').classList.toggle('on', s.camera_on);
  $('camErr').textContent = s.cam_error || '';

  $('words').innerHTML = s.words.length
    ? s.words.map(w => `<span class="chip">${w}</span>`).join('')
    : '<span class="stat">尚未辨識到詞彙，請對著鏡頭比手語</span>';

  $('sent').textContent = s.translating ? '翻譯中…' : (s.sentence || '—');
  $('by').textContent = s.sentence && !s.translating && s.sentence_by
                        ? '由' + s.sentence_by + '產生' : '';

  $('btnT').disabled = !s.words.length || s.translating;
  $('btnS').disabled = !s.sentence || s.translating;

  const st = s.status;
  let txt = st.has_hand ? `偵測到 ${st.hand_n} 隻手　` : '未偵測到手　';
  txt += st.has_face ? '臉：OK　' : '臉：無　';
  if (st.cooldown > 0) txt += `冷卻 ${st.cooldown}　`;
  else if (st.confirm_word) txt += `確認中：${st.confirm_word} ${st.confirm_pct}%　`;
  txt += `Buffer ${st.buffer_pct}%　Motion ${st.motion.toFixed(4)}　${st.fps} fps`;
  $('stat').textContent = txt;

  if (!dragging) {
    $('sc').value = s.conf_thresh;  $('vc').textContent = s.conf_thresh.toFixed(3);
    $('sf').value = s.confirm_frames; $('vf').textContent = s.confirm_frames;
    $('si').value = s.idle_to_translate; $('vi').textContent = s.idle_to_translate;
  }

  // 音訊版本變了才抓，避免重複播放
  if (s.has_audio && s.audio_seq !== lastAudio) {
    lastAudio = s.audio_seq;
    const au = $('au');
    au.src = '/audio?v=' + s.audio_seq;
    au.play().catch(() => {});   // 瀏覽器可能擋自動播放，忽略即可
  }

  $('hist').innerHTML = s.history.length
    ? s.history.map(r => `<div><code>${r.time||''}</code> ${(r.words||[]).join(' ')}
        → <b>${r.sentence||''}</b></div>`).join('')
    : '<span class="stat">尚無紀錄</span>';
  $('cls').textContent = '詞彙庫（' + s.classes.length + '）：' +
                         (s.classes.join('、') || '（模型未載入）');
}

tick();
setInterval(tick, 350);
</script>
</body>
</html>
"""


def prime_camera_permission():
    """macOS：授權請求只能在主執行緒發出。

    啟動伺服器前先在主執行緒開一次攝影機，第一次使用的人才會跳出系統授權視窗；
    之後背景的擷取執行緒就能直接開（靠上面的 SKIP_AUTH 跳過重複請求）。
    沒有這一步，新使用者只會看到「打不開攝影機」卻等不到授權視窗。
    """
    if sys.platform != "darwin":
        return
    prev = os.environ.get("OPENCV_AVFOUNDATION_SKIP_AUTH")
    os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "0"   # 這一次要讓它請求授權
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.read()
        cap.release()
    except Exception:
        pass
    finally:
        os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = prev if prev is not None else "1"


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser(description="手語即時翻譯網頁（FastAPI + MJPEG）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    prime_camera_permission()
    print(f"開啟 http://{args.host if args.host != '0.0.0.0' else '127.0.0.1'}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
