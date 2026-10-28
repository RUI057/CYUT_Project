"""台灣手語即時辨識 → 語義翻譯網頁（FastAPI + MJPEG 串流）。

流程：攝影機 → LSTM 辨識手語詞 → 累積詞彙序列 →
      本機語言模型轉成自然中文句子（可切換 Gemini，見 src/translate.py）→
      朗讀（gTTS，連不上網路時用本機語音，見 src/tts.py）。

執行：python web.py --open     （預設 http://127.0.0.1:8000，就緒後自動開瀏覽器）
      python web.py --port 9000 --host 0.0.0.0
      Mac 也可以直接雙擊「翻譯.command」。

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
import socket
import sys
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
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
from src import tts
from src.tts import synthesize
from src.video import open_camera, camera_help
from src.vocab import load_vocab, get_hint, get_demo_links
from src.version import APP_NAME, VERSION, ORG

RECORDS_PATH = "history/records.json"
SETTINGS_PATH = os.path.join("data", "web_settings.json")   # 網頁設定，重開後沿用（不進版控）
SETTING_LIMITS = {"conf_thresh": (0.3, 1.0), "confirm_frames": (5, 30),
                  "idle_to_translate": (10, 60)}
LAST_WORD_SEC = 2.5      # 剛確認的詞在畫面上提示多久
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
        self.status = {"has_hand": False, "hand_n": 0,
                       "confirm_word": "", "confirm_pct": 0, "buffer_pct": 0,
                       "motion": 0.0, "cooldown": 0, "fps": 0.0, "top": []}
        self.words = []
        self.sentence = ""
        self.sentence_by = ""
        self.audio = None
        self.audio_seq = 0          # 變動時前端才重新抓音訊，避免重複播放
        self.translating = False
        self.cam_error = ""
        self.history = load_history()
        self.idle_to_translate = 25
        self.auto_speak = True
        self.last_word, self.last_word_t = "", 0.0

        self.want_cam = threading.Event()
        self.stop = threading.Event()
        self.recognizer = camera.SignRecognizer()
        self.jobs = queue.Queue(maxsize=1)   # 翻譯工作；滿了就丟棄，不堆積
        self.defaults = {**self.current_settings(),
                         "conf_thresh": float(self.recognizer.conf_thresh)}   # 恢復預設用，不四捨五入
        self.load_settings()

    # ── 設定：存在 data/web_settings.json，重開後沿用 ──
    def current_settings(self):
        return {"conf_thresh": round(float(self.recognizer.conf_thresh), 3),
                "confirm_frames": int(self.recognizer.confirm_frames),
                "idle_to_translate": int(self.idle_to_translate),
                "auto_speak": bool(self.auto_speak)}

    def apply_settings(self, d):
        """套用設定（呼叫端要持有 lock）；數值會夾在合理範圍內。"""
        def clamp(key, v):
            lo, hi = SETTING_LIMITS[key]
            return max(lo, min(hi, v))
        if d.get("conf_thresh") is not None:
            self.recognizer.conf_thresh = clamp("conf_thresh", float(d["conf_thresh"]))
        if d.get("confirm_frames") is not None:
            self.recognizer.confirm_frames = int(clamp("confirm_frames", int(d["confirm_frames"])))
        if d.get("idle_to_translate") is not None:
            self.idle_to_translate = int(clamp("idle_to_translate", int(d["idle_to_translate"])))
        if d.get("auto_speak") is not None:
            self.auto_speak = bool(d["auto_speak"])

    def save_settings(self):
        with self.lock:
            data = self.current_settings()
        data["model_sig"] = _model_sig()
        try:
            os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
            tmp = SETTINGS_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, SETTINGS_PATH)
        except OSError as e:
            print(f"[設定] 無法儲存：{e}")

    def load_settings(self):
        try:
            with open(SETTINGS_PATH, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        # 信心門檻是訓練時校準的：模型重訓過就改用新模型的校準值，不沿用舊的
        if d.get("model_sig") != _model_sig():
            d.pop("conf_thresh", None)
        with self.lock:
            self.apply_settings(d)

    def reset_settings(self):
        with self.lock:
            self.apply_settings(self.defaults)
        try:
            os.remove(SETTINGS_PATH)
        except OSError:
            pass

    # ── 給端點用的小工具 ──
    def snapshot(self):
        with self.lock:
            recent = time.time() - self.last_word_t < LAST_WORD_SEC
            snap = {
                "camera_on": self.want_cam.is_set(),
                "cam_error": self.cam_error,
                "words": list(self.words),
                "sentence": self.sentence,
                "sentence_by": self.sentence_by,
                "translating": self.translating,
                "audio_seq": self.audio_seq,
                "has_audio": self.audio is not None,
                "status": dict(self.status),
                **self.current_settings(),
                "last_word": self.last_word if recent else "",
                "history": self.history[:30],
            }
        snap.update(model={"ready": camera.is_ready(), "n": len(camera.get_classes())},
                    llm=translate.backend_status(), tts=tts.backend_status())
        return snap

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


def _model_sig():
    """模型的識別：詞數 + 校準過的信心門檻。重訓後會變，存檔的信心門檻就不再沿用。"""
    return f"{len(camera.get_classes())}:{camera.default_conf():.4f}"


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
            camera.draw_landmarks(frame, state["results"])

            # 確認到新詞 → 加入序列（連續同詞只算一次）
            if state["confirmed"]:
                with hub.lock:
                    if not hub.words or hub.words[-1] != state["confirmed"]:
                        hub.words.append(state["confirmed"])
                    hub.last_word, hub.last_word_t = state["confirmed"], time.time()

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
                    "has_hand": state["has_hand"],
                    "hand_n": state["hand_n"], "confirm_word": state["confirm_word"],
                    "confirm_pct": state["confirm_pct"], "buffer_pct": state["buffer_pct"],
                    "motion": round(float(state["motion"]), 4),
                    "cooldown": state["cooldown"], "fps": round(fps, 1),
                    "top": state.get("top", []),
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
@asynccontextmanager
async def lifespan(_app):
    """啟動時開擷取／翻譯執行緒，關閉時收掉（取代已棄用的 on_event）。"""
    threads = []
    for fn in (capture_loop, worker_loop):
        t = threading.Thread(target=fn, args=(HUB,), daemon=True)
        t.start()
        threads.append(t)
    threading.Thread(target=translate.warmup, daemon=True).start()  # 預熱，第一次翻譯不用等
    try:
        yield
    finally:
        HUB.stop.set()          # /stream 的產生器與兩條執行緒都靠它結束
        HUB.want_cam.clear()
        for t in threads:
            t.join(timeout=2)


app = FastAPI(title="手語即時翻譯", lifespan=lifespan)


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
    return Response(audio, media_type=tts.mime_of(audio),
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


class SpeakReq(BaseModel):
    text: str | None = None      # 不給就念目前的句子；給了就念這段（翻譯紀錄的「重播」）


@app.post("/speak")
async def do_speak(req: SpeakReq | None = None):
    text = (req.text or "").strip()[:200] if req else ""
    with HUB.lock:
        sentence = text or HUB.sentence
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


class WordReq(BaseModel):
    index: int


@app.post("/words/remove")
async def remove_word(req: WordReq):
    """刪掉認錯的某一個詞（詞彙列上的 ×）。"""
    with HUB.lock:
        ok = 0 <= req.index < len(HUB.words)
        if ok:
            del HUB.words[req.index]
    return {"ok": ok}


@app.post("/words/pop")
async def pop_word():
    """刪掉最後一個詞（⌫）。"""
    with HUB.lock:
        ok = bool(HUB.words)
        if ok:
            HUB.words.pop()
    return {"ok": ok}


class Settings(BaseModel):
    conf_thresh: float | None = None
    confirm_frames: int | None = None
    idle_to_translate: int | None = None
    auto_speak: bool | None = None


@app.post("/settings")
async def set_settings(s: Settings):
    with HUB.lock:
        HUB.apply_settings(s.model_dump())
    HUB.save_settings()
    return {"ok": True}


@app.post("/settings/reset")
async def reset_settings():
    HUB.reset_settings()
    return {"ok": True}


@app.get("/vocab")
async def get_vocab():
    """手語字典：詞彙表分組 + 模型認不認得 + 示範影片 + 比法提示。"""
    known = set(camera.get_classes())
    links = get_demo_links()
    groups = [{"name": g["name"],
               "words": [{"w": w, "ok": w in known, "url": links.get(w, ""), "hint": get_hint(w)}
                         for w in g["labels"]]}
              for g in load_vocab()["groups"]]
    total = sum(len(g["words"]) for g in groups)
    return {"groups": groups, "n_total": total,
            "n_known": sum(1 for g in groups for x in g["words"] if x["ok"])}


@app.get("/", response_class=HTMLResponse)
async def index():
    warn = "" if camera.is_ready() else (
        '<div class="banner">找不到辨識模型（<code>data/model.pkl</code>），畫面可以開，但不會辨識。'
        '請先在選單選「2 訓練模型」，或執行 <code>python train_model.py</code>。</div>')
    return (PAGE.replace("<!--WARN-->", warn)
                .replace("<!--APP-->", APP_NAME)
                .replace("<!--VER-->", VERSION)
                .replace("<!--ORG-->", ORG))


PAGE = r"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title><!--APP--></title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%231f6feb'/%3E%3Ctext x='32' y='44' font-size='34' text-anchor='middle' fill='white' font-family='sans-serif'%3E%E6%89%8B%3C/text%3E%3C/svg%3E">
<style>
  :root {
    --bg:#0d1017; --panel:#151a23; --panel2:#1b2130; --line:#283042; --fg:#e8ebf2;
    --muted:#9aa4b6; --dim:#6b7488; --accent:#3b82f6; --accent2:#2563eb;
    --ok:#22c55e; --warn:#f59e0b; --bad:#ef4444; --radius:14px;
  }
  * { box-sizing:border-box; }
  html, body { height:100%; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font-family:-apple-system,BlinkMacSystemFont,"PingFang TC","Noto Sans TC","Microsoft JhengHei",sans-serif;
         -webkit-font-smoothing:antialiased; display:flex; flex-direction:column; }
  button { font:inherit; color:inherit; }
  kbd { font:11px ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--muted);
        border:1px solid var(--line); border-bottom-width:2px; border-radius:5px; padding:1px 5px; margin-left:6px; }

  /* 頂部 */
  .top { display:flex; align-items:center; gap:16px; padding:12px 22px; border-bottom:1px solid var(--line);
         background:rgba(13,16,23,.9); position:sticky; top:0; z-index:5; backdrop-filter:blur(8px); flex-wrap:wrap; }
  .brand { display:flex; align-items:center; gap:12px; margin-right:auto; }
  .logo { width:38px; height:38px; border-radius:11px; display:grid; place-items:center; font-weight:700; font-size:20px;
          background:linear-gradient(135deg,#3b82f6,#8b5cf6); box-shadow:0 4px 14px rgba(59,130,246,.35); }
  .brand h1 { margin:0; font-size:18px; letter-spacing:.5px; }
  .brand p { margin:2px 0 0; font-size:12px; color:var(--muted); }
  .pills { display:flex; gap:8px; flex-wrap:wrap; }
  .pill { display:flex; align-items:center; gap:6px; font-size:12px; color:var(--muted); background:var(--panel);
          border:1px solid var(--line); border-radius:999px; padding:5px 11px; white-space:nowrap; }
  .pill i { width:7px; height:7px; border-radius:50%; background:var(--dim); }
  .pill.ok i { background:var(--ok); } .pill.warn i { background:var(--warn); } .pill.bad i { background:var(--bad); }
  .ghost { background:transparent; border:1px solid var(--line); border-radius:9px; padding:7px 12px; cursor:pointer; font-size:13px; }
  .ghost:hover { border-color:var(--accent); }

  .banner { margin:14px 22px 0; padding:12px 16px; border-radius:10px; background:rgba(245,158,11,.12);
            border:1px solid rgba(245,158,11,.4); color:#fcd34d; font-size:14px; }
  .banner code { color:#fde68a; }

  /* 主畫面 */
  main { flex:1; display:grid; grid-template-columns:minmax(0,1.55fr) minmax(330px,1fr); gap:20px; padding:20px 22px; }
  @media (max-width:980px) { main { grid-template-columns:1fr; } }
  .stage { display:flex; flex-direction:column; gap:12px; min-width:0; }
  .video { position:relative; border-radius:var(--radius); overflow:hidden; background:#000;
           border:1px solid var(--line); aspect-ratio:4/3; }
  .video img { width:100%; height:100%; object-fit:contain; display:block; }
  .hud { position:absolute; left:0; right:0; bottom:0; padding:18px; display:flex; flex-direction:column; gap:10px;
         background:linear-gradient(transparent, rgba(0,0,0,.75)); pointer-events:none; }
  .hint { align-self:flex-start; font-size:20px; font-weight:600; padding:8px 16px; border-radius:12px;
          background:rgba(15,18,26,.78); border:1px solid rgba(255,255,255,.08); transition:all .2s; }
  .hint:empty { display:none; }
  .hint.ok { background:rgba(34,197,94,.9); color:#04140a; border-color:transparent; }
  .hint.busy { background:rgba(59,130,246,.85); border-color:transparent; }
  .live { display:none; gap:12px; align-items:center; }
  .live.on { display:flex; }
  .live b { font-size:26px; min-width:3em; }
  .bar { flex:1; height:8px; background:rgba(255,255,255,.15); border-radius:9px; overflow:hidden; max-width:320px; }
  .bar i { display:block; height:100%; width:0; background:var(--accent); transition:width .15s; border-radius:9px; }
  .cands { display:flex; gap:6px; flex-wrap:wrap; min-height:26px; }
  .cand { font-size:13px; padding:3px 10px; border-radius:999px; background:rgba(15,18,26,.75); color:var(--muted); }
  .cand:first-child { color:var(--fg); border:1px solid rgba(59,130,246,.6); }
  .controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  .camerr { color:#fca5a5; font-size:13px; white-space:pre-line; }

  .btn { border:1px solid var(--line); background:var(--panel2); border-radius:10px; padding:11px 14px;
         cursor:pointer; font-size:14px; display:inline-flex; align-items:center; justify-content:center; gap:4px; }
  .btn:hover:not(:disabled) { border-color:var(--accent); }
  .btn:disabled { opacity:.38; cursor:default; }
  .btn.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  .btn.primary:hover:not(:disabled) { background:var(--accent2); }
  .btn.primary kbd { color:#dbeafe; border-color:rgba(255,255,255,.35); }
  .btn.big { padding:13px 22px; font-size:16px; font-weight:600; }
  .btn.on { background:#1f2937; border-color:var(--bad); }

  /* 側欄 */
  .side { display:flex; flex-direction:column; gap:16px; min-width:0; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); padding:16px; }
  .card h2 { margin:0 0 4px; font-size:15px; }
  .card .sub { color:var(--dim); font-size:12px; margin:0 0 12px; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; min-height:38px; align-items:center; }
  .chip { display:inline-flex; align-items:center; gap:4px; background:rgba(59,130,246,.16); color:#cfe0ff;
          border:1px solid rgba(59,130,246,.45); padding:6px 6px 6px 13px; border-radius:999px; font-size:17px; }
  .chip button { border:0; background:transparent; color:#93b4f5; cursor:pointer; font-size:15px; line-height:1;
                 width:22px; height:22px; border-radius:50%; }
  .chip button:hover { background:rgba(255,255,255,.12); color:#fff; }
  .empty { color:var(--dim); font-size:14px; }
  .sentence { font-size:26px; line-height:1.45; margin:16px 0 4px; min-height:38px; font-weight:600; }
  .sentence.placeholder { font-size:15px; color:var(--dim); font-weight:400; }
  .by { color:var(--dim); font-size:12px; min-height:16px; }
  .actions { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-top:14px; }
  .actions .btn { padding:10px 6px; font-size:13px; }
  .actions kbd { display:none; }
  @media (min-width:1280px) { .actions kbd { display:inline; } }

  .tabs { display:flex; gap:4px; margin:-4px 0 12px; border-bottom:1px solid var(--line); }
  .tabs button { background:none; border:0; padding:8px 12px; cursor:pointer; color:var(--muted); font-size:14px;
                 border-bottom:2px solid transparent; margin-bottom:-1px; }
  .tabs button.on { color:var(--fg); border-bottom-color:var(--accent); }
  .pane { display:none; } .pane.on { display:block; }
  .hist { max-height:300px; overflow:auto; display:flex; flex-direction:column; gap:2px; }
  .hrow { display:grid; grid-template-columns:auto 1fr auto; gap:10px; align-items:center; padding:8px 4px;
          border-bottom:1px solid var(--line); font-size:14px; }
  .hrow time { color:var(--dim); font-size:12px; font-variant-numeric:tabular-nums; }
  .hrow small { display:block; color:var(--dim); font-size:12px; margin-top:2px; }
  .hrow button { border:1px solid var(--line); background:none; border-radius:8px; padding:4px 9px; cursor:pointer; font-size:12px; color:var(--muted); }
  .hrow button:hover { color:var(--fg); border-color:var(--accent); }
  .search { width:100%; padding:9px 12px; border-radius:9px; border:1px solid var(--line); background:var(--panel2);
            color:var(--fg); font-size:14px; margin-bottom:10px; }
  .dict { max-height:340px; overflow:auto; }
  .dgroup { margin-bottom:12px; }
  .dgroup h3 { margin:0 0 6px; font-size:12px; color:var(--muted); font-weight:600; }
  .dwords { display:flex; flex-wrap:wrap; gap:6px; }
  .dw { border:1px solid var(--line); background:var(--panel2); border-radius:8px; padding:5px 10px; font-size:14px; cursor:pointer; }
  .dw:hover { border-color:var(--accent); }
  .dw.off { color:var(--dim); border-style:dashed; }
  .legend { font-size:12px; color:var(--dim); margin:-2px 0 10px; }
  .set { margin-bottom:16px; }
  .set label { display:flex; justify-content:space-between; font-size:14px; }
  .set label span { color:var(--muted); font-variant-numeric:tabular-nums; }
  .set p { margin:2px 0 6px; font-size:12px; color:var(--dim); }
  .set input[type=range] { width:100%; accent-color:var(--accent); }
  .switch { display:flex; align-items:center; gap:10px; font-size:14px; cursor:pointer; margin-bottom:16px; }
  .switch input { width:18px; height:18px; accent-color:var(--accent); }
  .tech { font-size:12px; color:var(--dim); line-height:1.7; border-top:1px solid var(--line); padding-top:10px; margin-top:4px;
          font-variant-numeric:tabular-nums; }

  footer { padding:14px 22px 18px; color:var(--dim); font-size:12px; display:flex; gap:14px; flex-wrap:wrap; justify-content:space-between; }

  /* 展示模式：只留畫面與大字幕，適合投影或展場螢幕 */
  .caption { display:none; }
  body.present .top, body.present .side, body.present footer, body.present .banner, body.present .controls { display:none; }
  body.present main { grid-template-columns:1fr; padding:16px; }
  body.present .video { aspect-ratio:auto; height:calc(100vh - 210px); }
  body.present .caption { display:block; text-align:center; font-size:46px; font-weight:700; line-height:1.3;
                          padding:14px 20px; min-height:84px; }
  body.present .caption small { display:block; font-size:22px; color:var(--muted); font-weight:500; margin-top:4px; }
  body.present .hint { font-size:24px; }
  .exitp { display:none; position:fixed; top:14px; right:16px; z-index:9; }
  body.present .exitp { display:block; }

  /* 說明視窗、提示訊息 */
  .modal { position:fixed; inset:0; background:rgba(0,0,0,.6); display:none; place-items:center; z-index:20; padding:20px; }
  .modal.on { display:grid; }
  .dialog { background:var(--panel); border:1px solid var(--line); border-radius:18px; max-width:560px; width:100%;
            padding:26px; box-shadow:0 20px 60px rgba(0,0,0,.5); max-height:90vh; overflow:auto; }
  .dialog h2 { margin:0 0 6px; font-size:22px; }
  .dialog > p { color:var(--muted); margin:0 0 18px; font-size:14px; }
  .steps { display:flex; flex-direction:column; gap:12px; margin-bottom:18px; }
  .step { display:flex; gap:14px; align-items:flex-start; }
  .step b { flex:none; width:30px; height:30px; border-radius:50%; background:var(--accent); display:grid; place-items:center; }
  .step div { font-size:15px; line-height:1.55; }
  .step small { display:block; color:var(--dim); font-size:13px; }
  .keys { display:grid; grid-template-columns:auto 1fr auto 1fr; gap:8px 10px; font-size:13px; color:var(--muted);
          border-top:1px solid var(--line); padding-top:14px; margin-bottom:18px; align-items:center; }
  .keys kbd { margin:0; justify-self:start; }
  .privacy { font-size:12px; color:var(--dim); margin:0 0 18px; line-height:1.6; }
  .toast { position:fixed; left:50%; bottom:28px; transform:translate(-50%,20px); opacity:0; pointer-events:none;
           background:#111827; border:1px solid var(--line); padding:11px 18px; border-radius:12px; font-size:14px;
           transition:all .25s; z-index:30; max-width:90vw; }
  .toast.show { opacity:1; transform:translate(-50%,0); }
  .toast.bad { border-color:var(--bad); }
</style>
</head>
<body>
<header class="top">
  <div class="brand">
    <div class="logo">手</div>
    <div><h1><!--APP--></h1><p>台灣手語 → 中文句子 → 語音</p></div>
  </div>
  <div class="pills">
    <span class="pill" id="pModel" title=""><i></i><span>辨識模型</span></span>
    <span class="pill" id="pLlm" title=""><i></i><span>語義修正</span></span>
    <span class="pill" id="pTts" title=""><i></i><span>朗讀</span></span>
  </div>
  <button class="ghost" data-act="present" title="全螢幕，只顯示畫面與大字幕">展示模式<kbd>F</kbd></button>
  <button class="ghost" data-act="help">使用說明<kbd>?</kbd></button>
</header>
<!--WARN-->

<main>
  <section class="stage">
    <div class="video">
      <img id="cam" src="/stream" alt="攝影機畫面">
      <div class="hud">
        <div class="cands" id="cands"></div>
        <div class="live" id="live"><b id="liveWord"></b><div class="bar"><i id="liveBar"></i></div></div>
        <div class="hint" id="hint"></div>
      </div>
    </div>
    <div class="caption" id="caption"></div>
    <div class="controls">
      <button class="btn primary big" id="camBtn" data-act="cam">啟動攝影機<kbd>C</kbd></button>
      <span class="camerr" id="camErr"></span>
    </div>
  </section>

  <aside class="side">
    <div class="card">
      <h2>這句話</h2>
      <p class="sub">一個詞比完稍停一下；整段比完把手放下，會自動翻譯並朗讀</p>
      <div class="chips" id="words"></div>
      <div class="sentence placeholder" id="sent"></div>
      <div class="by" id="by"></div>
      <div class="actions">
        <button class="btn primary" id="btnT" data-act="translate">翻譯<kbd>↵</kbd></button>
        <button class="btn" id="btnS" data-act="speak">朗讀<kbd>S</kbd></button>
        <button class="btn" id="btnU" data-act="undo">刪最後一詞<kbd>⌫</kbd></button>
        <button class="btn" id="btnC" data-act="clear">清除<kbd>Esc</kbd></button>
      </div>
    </div>

    <div class="card">
      <div class="tabs" id="tabs">
        <button class="on" data-tab="hist">翻譯紀錄</button>
        <button data-tab="dict">手語字典</button>
        <button data-tab="set">設定</button>
      </div>
      <div class="pane on" id="pane-hist"><div class="hist" id="hist"></div></div>
      <div class="pane" id="pane-dict">
        <input class="search" id="q" placeholder="搜尋詞彙，例如：謝謝" autocomplete="off">
        <p class="legend" id="legend"></p>
        <div class="dict" id="dict"></div>
      </div>
      <div class="pane" id="pane-set">
        <label class="switch"><input type="checkbox" id="autoSpeak">翻譯完自動朗讀</label>
        <div class="set">
          <label>辨識嚴格度<span id="vc"></span></label>
          <p>越高越不容易認錯，但比較難觸發；預設值是訓練時依資料校準的</p>
          <input type="range" id="sc" min="0.3" max="1" step="0.005" data-key="conf_thresh">
        </div>
        <div class="set">
          <label>確認速度（連續幾幀同一個詞）<span id="vf"></span></label>
          <p>越少出詞越快，但比較容易誤判</p>
          <input type="range" id="sf" min="5" max="30" step="1" data-key="confirm_frames">
        </div>
        <div class="set">
          <label>自動翻譯等待（手放下幾幀）<span id="vi"></span></label>
          <p>手離開畫面多久之後，把累積的詞翻成句子</p>
          <input type="range" id="si" min="10" max="60" step="5" data-key="idle_to_translate">
        </div>
        <button class="btn" data-act="reset">恢復預設</button>
        <div class="tech" id="tech"></div>
      </div>
    </div>
  </aside>
</main>

<footer>
  <span><!--APP--> v<!--VER--> · <!--ORG--></span>
  <span>攝影機畫面只在這台電腦上處理，不會上傳</span>
</footer>

<button class="btn exitp" data-act="present">離開展示模式<kbd>F</kbd></button>

<div class="modal" id="guide">
  <div class="dialog">
    <h2>歡迎使用<!--APP--></h2>
    <p>對著鏡頭比台灣手語，系統會即時辨識詞彙、組成通順的中文句子並念出來。</p>
    <div class="steps">
      <div class="step"><b>1</b><div>按「啟動攝影機」<small>上半身和雙手都要在畫面裡，光線越亮越好。第一次使用請允許瀏覽器與系統使用攝影機。</small></div></div>
      <div class="step"><b>2</b><div>一個詞比完，稍停一下<small>畫面出現綠色的「✓ 詞」就代表認到了。認錯可以按詞旁邊的 ×，或按 ⌫ 刪掉最後一個。</small></div></div>
      <div class="step"><b>3</b><div>整段比完，把手放下<small>系統會自動把詞組成句子並朗讀，也可以直接按「翻譯」。</small></div></div>
    </div>
    <div class="keys">
      <kbd>C</kbd><span>開關攝影機</span><kbd>↵</kbd><span>翻譯</span>
      <kbd>S</kbd><span>朗讀</span><kbd>⌫</kbd><span>刪最後一詞</span>
      <kbd>Esc</kbd><span>清除</span><kbd>F</kbd><span>展示模式</span>
    </div>
    <p class="privacy">攝影機畫面只在這台電腦上處理、不會上傳。使用線上語音朗讀時，句子的文字會傳給 Google 合成語音；
      連不上網路時自動改用電腦內建的語音。不知道某個詞怎麼比？到「手語字典」點那個詞看示範影片。</p>
    <button class="btn primary big" data-act="closeGuide" style="width:100%">開始使用</button>
  </div>
</div>

<div class="toast" id="toast"></div>
<audio id="au" style="display:none"></audio>

<script>
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} },
};
let S = null, firstTick = true, lastAudio = -1, wantPlay = false, dragging = false;
let lastErr = '', lastBy = '', histKey = '', vocab = null;

function toast(msg, kind) {
  const t = $('toast');
  t.textContent = msg; t.className = 'toast show ' + (kind || '');
  clearTimeout(toast.timer); toast.timer = setTimeout(() => t.className = 'toast', 3800);
}

async function api(url, body) {
  try {
    const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'},
                                body: JSON.stringify(body || {})});
    const j = await r.json().catch(() => ({}));
    tick();
    return j;
  } catch (e) {
    toast('連不上翻譯程式，請確認它還在執行', 'bad');
    return {};
  }
}

const actions = {
  cam:        () => api('/camera', {on: !(S && S.camera_on)}),
  translate:  async () => { const j = await api('/translate'); if (j.ok === false && S && !S.words.length) toast('還沒有辨識到詞彙'); },
  speak:      () => { wantPlay = true; return api('/speak'); },
  undo:       () => api('/words/pop'),
  clear:      () => api('/clear'),
  reset:      async () => { await api('/settings/reset'); toast('已恢復預設設定'); },
  present:    togglePresent,
  help:       () => $('guide').classList.add('on'),
  closeGuide: () => { $('guide').classList.remove('on'); store.set('guideSeen', '1'); },
};
document.addEventListener('click', e => {
  const b = e.target.closest('[data-act]');
  if (b && !b.disabled) actions[b.dataset.act]();
  const x = e.target.closest('[data-rm]');
  if (x) api('/words/remove', {index: +x.dataset.rm});
  const say = e.target.closest('[data-say]');
  if (say) { wantPlay = true; api('/speak', {text: say.dataset.say}); }
  const w = e.target.closest('[data-url]');
  if (w && w.dataset.url) window.open(w.dataset.url, '_blank', 'noopener');
  const tab = e.target.closest('[data-tab]');
  if (tab) {
    document.querySelectorAll('#tabs button').forEach(t => t.classList.toggle('on', t === tab));
    document.querySelectorAll('.pane').forEach(p => p.classList.toggle('on', p.id === 'pane-' + tab.dataset.tab));
    if (tab.dataset.tab === 'dict') loadVocab();
  }
});

document.addEventListener('keydown', e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return;
  if ($('guide').classList.contains('on')) {
    if (e.key === 'Escape' || e.key === 'Enter') { e.preventDefault(); actions.closeGuide(); }
    return;
  }
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  const map = {c:'cam', Enter:'translate', s:'speak', Backspace:'undo', Escape:'clear', f:'present', '?':'help', h:'help'};
  const name = map[k];
  if (!name) return;
  if (name === 'clear' && document.body.classList.contains('present')) return;
  e.preventDefault();
  actions[name]();
});

function togglePresent() {
  const on = !document.body.classList.contains('present');
  document.body.classList.toggle('present', on);
  if (on && document.documentElement.requestFullscreen) document.documentElement.requestFullscreen().catch(() => {});
  if (!on && document.fullscreenElement) document.exitFullscreen().catch(() => {});
}
document.addEventListener('fullscreenchange', () => {
  if (!document.fullscreenElement) document.body.classList.remove('present');
});

// 設定：拖動時只更新數字，放開才送出（避免一路狂送）
const fmt = {conf_thresh: v => (+v).toFixed(3), confirm_frames: v => v, idle_to_translate: v => v};
const label = {sc:'vc', sf:'vf', si:'vi'};
for (const id of ['sc', 'sf', 'si']) {
  const el = $(id);
  el.addEventListener('input', () => { dragging = true; $(label[id]).textContent = fmt[el.dataset.key](el.value); });
  el.addEventListener('change', async () => { await api('/settings', {[el.dataset.key]: +el.value}); dragging = false; });
}
$('autoSpeak').addEventListener('change', e => api('/settings', {auto_speak: e.target.checked}));

function pill(id, text, kind, title) {
  const p = $(id);
  p.className = 'pill ' + kind; p.title = title || text;
  p.querySelector('span').textContent = text;
}

function render() {
  const st = S.status;
  // 頂部狀態
  pill('pModel', S.model.ready ? `辨識模型 · ${S.model.n} 詞` : '辨識模型未載入', S.model.ready ? 'ok' : 'bad');
  const llmKind = /載入中|未載入/.test(S.llm) ? 'warn' : /失敗/.test(S.llm) ? 'bad' : 'ok';
  pill('pLlm', '語義修正 · ' + (/載入中/.test(S.llm) ? '載入中' : /失敗/.test(S.llm) ? '載入失敗' : /Gemini/.test(S.llm) ? '雲端' : '本機'),
       llmKind, S.llm);
  pill('pTts', '朗讀 · ' + (/（離線）/.test(S.tts) ? '本機語音' : '線上'),
       /找不到/.test(S.tts) ? 'warn' : 'ok', S.tts);

  // 攝影機按鈕
  $('camBtn').firstChild.textContent = S.camera_on ? '關閉攝影機' : '啟動攝影機';
  $('camBtn').classList.toggle('on', S.camera_on);
  $('camBtn').classList.toggle('primary', !S.camera_on);
  $('camErr').textContent = S.cam_error || '';
  if (S.cam_error && S.cam_error !== lastErr) toast('攝影機打不開，請看按鈕旁的說明', 'bad');
  lastErr = S.cam_error || '';

  // 畫面上的即時提示
  let hint = '', cls = '';
  if (!S.camera_on) hint = '';            // 攝影機沒開時，畫面中央的佔位圖已經有說明
  else if (S.translating) { hint = '翻譯中…'; cls = 'busy'; }
  else if (S.last_word) { hint = '✓ ' + S.last_word; cls = 'ok'; }
  else if (!st.has_hand) hint = '把手放進畫面';
  else if (st.cooldown > 0) hint = '請比下一個詞';
  else if (st.confirm_word) hint = '辨識中…';
  else hint = '比出手語，每個詞之間稍停一下';
  $('hint').textContent = hint; $('hint').className = 'hint ' + cls;

  const showLive = S.camera_on && st.confirm_word && !S.last_word;
  $('live').classList.toggle('on', !!showLive);
  if (showLive) { $('liveWord').textContent = st.confirm_word; $('liveBar').style.width = st.confirm_pct + '%'; }
  const top = (S.camera_on && st.has_hand && !S.last_word) ? (st.top || []) : [];
  $('cands').innerHTML = top.map(([w, p]) => `<span class="cand">${esc(w)} ${Math.round(p * 100)}%</span>`).join('');

  // 這句話
  $('words').innerHTML = S.words.length
    ? S.words.map((w, i) => `<span class="chip">${esc(w)}<button data-rm="${i}" title="刪掉這個詞">×</button></span>`).join('')
    : '<span class="empty">還沒有詞彙，對著鏡頭比手語</span>';
  const sent = $('sent');
  if (S.translating) { sent.textContent = '翻譯中…'; sent.className = 'sentence placeholder'; }
  else if (S.sentence) { sent.textContent = S.sentence; sent.className = 'sentence'; }
  else { sent.textContent = '翻譯好的句子會出現在這裡'; sent.className = 'sentence placeholder'; }
  $('by').textContent = S.sentence && !S.translating && S.sentence_by ? '由' + S.sentence_by + '產生' : '';
  if (S.sentence_by && S.sentence_by !== lastBy && S.sentence_by.startsWith('失敗')) toast('翻譯失敗：' + S.sentence_by, 'bad');
  lastBy = S.sentence_by;

  $('btnT').disabled = !S.words.length || S.translating;
  $('btnS').disabled = !S.sentence || S.translating;
  $('btnU').disabled = !S.words.length;
  $('btnC').disabled = !S.words.length && !S.sentence;

  // 展示模式的大字幕
  $('caption').innerHTML = S.sentence
    ? esc(S.sentence) + (S.words.length ? `<small>${esc(S.words.join(' '))}</small>` : '')
    : (S.words.length ? esc(S.words.join(' ')) : `<small>${esc(hint || '按 C 啟動攝影機')}</small>`);

  // 設定
  if (!dragging) {
    $('sc').value = S.conf_thresh; $('vc').textContent = (+S.conf_thresh).toFixed(3);
    $('sf').value = S.confirm_frames; $('vf').textContent = S.confirm_frames;
    $('si').value = S.idle_to_translate; $('vi').textContent = S.idle_to_translate;
  }
  $('autoSpeak').checked = S.auto_speak;
  $('tech').innerHTML = `技術資訊：${esc(S.llm)}；朗讀：${esc(S.tts)}<br>`
    + `偵測到 ${st.hand_n} 隻手 · Buffer ${st.buffer_pct}% · Motion ${(+st.motion).toFixed(4)} · ${st.fps} fps`;

  // 朗讀：音訊版本變了才抓。剛打開頁面時不重播上一句
  if (firstTick) lastAudio = S.audio_seq;
  if (S.has_audio && S.audio_seq !== lastAudio) {
    lastAudio = S.audio_seq;
    if (S.auto_speak || wantPlay) {
      const au = $('au');
      au.src = '/audio?v=' + S.audio_seq;
      au.play().catch(() => toast('瀏覽器擋住了自動播放，請先點一下頁面再按「朗讀」'));
    }
    wantPlay = false;
  }

  // 翻譯紀錄：有變才重畫，避免捲動位置跳掉
  const key = S.history.length + '|' + (S.history[0] ? S.history[0].time + S.history[0].sentence : '');
  if (key !== histKey) {
    histKey = key;
    $('hist').innerHTML = S.history.length
      ? S.history.map(r => `<div class="hrow"><time>${esc(r.time)}</time>
          <div>${esc(r.sentence)}<small>${esc((r.words || []).join(' '))}</small></div>
          <button data-say="${esc(r.sentence)}" title="再念一次">▶ 重播</button></div>`).join('')
      : '<span class="empty">翻譯過的句子會留在這裡</span>';
  }
  firstTick = false;
}

async function tick() {
  try { S = await (await fetch('/state')).json(); } catch (e) { return; }
  render();
}

// 手語字典
async function loadVocab() {
  if (!vocab) {
    try { vocab = await (await fetch('/vocab')).json(); } catch (e) { return; }
  }
  $('legend').textContent = `共 ${vocab.n_total} 個詞，目前認得 ${vocab.n_known} 個（虛線框 = 還在收集資料）。點詞彙看示範影片。`;
  const q = $('q').value.trim();
  $('dict').innerHTML = vocab.groups.map(g => {
    const ws = g.words.filter(x => !q || x.w.includes(q));
    if (!ws.length) return '';
    return `<div class="dgroup"><h3>${esc(g.name)}</h3><div class="dwords">` + ws.map(x =>
      `<span class="dw ${x.ok ? '' : 'off'}" data-url="${esc(x.url)}"
         title="${esc((x.ok ? '' : '還在收集資料，暫時認不出來。') + (x.hint ? '比法：' + x.hint : '') + (x.url ? '（點一下看示範影片）' : ''))}">${esc(x.w)}</span>`
    ).join('') + '</div></div>';
  }).join('') || '<span class="empty">找不到這個詞</span>';
}
$('q').addEventListener('input', loadVocab);

if (!store.get('guideSeen')) $('guide').classList.add('on');
tick();
setInterval(tick, 300);
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


# ── 啟動：連接埠被佔用時的處理、就緒後自動開瀏覽器 ──
def _port_free(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _already_running(port):
    """這個連接埠上跑的是不是本程式（重複雙擊啟動時，直接打開已經在跑的那一個）。"""
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=1.5) as r:
            return r.status == 200 and b'"camera_on"' in r.read(4096)
    except Exception:
        return False


def _open_when_ready(url, port, timeout=90):
    """伺服器真的開始接連線之後才開瀏覽器（模型載入慢的電腦固定等 2 秒會開到錯誤頁）。"""
    def run():
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.3)
        webbrowser.open(url)
    threading.Thread(target=run, daemon=True).start()


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser(description=f"{APP_NAME}（FastAPI + MJPEG）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open", action="store_true", help="伺服器就緒後自動用瀏覽器開啟")
    args = ap.parse_args()

    port = args.port
    if not _port_free(args.host, port):
        if _already_running(port):
            url = f"http://127.0.0.1:{port}"
            print(f"{APP_NAME} 已經在執行：{url}")
            if args.open:
                webbrowser.open(url)
            sys.exit(0)
        port = next((p for p in range(port + 1, port + 20) if _port_free(args.host, p)), None)
        if port is None:
            print(f"❌ 連接埠 {args.port}～{args.port + 19} 都被佔用，請用 --port 指定別的連接埠。")
            sys.exit(1)
        print(f"連接埠 {args.port} 被其他程式佔用，改用 {port}")

    url = f"http://127.0.0.1:{port}"
    prime_camera_permission()
    print(f"{APP_NAME} v{VERSION}：{url}")
    if args.open:
        _open_when_ready(url, port)
    uvicorn.run(app, host=args.host, port=port, log_level="warning")
