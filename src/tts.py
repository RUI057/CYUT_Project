"""文字轉語音：線上 gTTS 優先，連不上時自動改用本機語音（離線）。

展場的網路常常不穩，朗讀不能因為連不上 Google 就沒聲音。
- 線上：gTTS（音質較好，需要網路），回傳 mp3。
- 離線：macOS 內建的 `say`（預設台灣中文語音 Meijia），回傳 wav。目前只支援 macOS。
- 線上失敗後 ONLINE_RETRY_SEC 秒內直接走離線，不要每一句都先等逾時；時間到再試一次線上。

環境變數：
  TTS_BACKEND  auto（預設）｜online（只用 gTTS）｜offline（只用本機語音，展場建議）
  TTS_VOICE    本機語音名稱（預設自動挑台灣中文：Meijia 優先）；列表見 `say -v '?'`

synthesize 回傳的位元組格式依後端而定，播放時用 mime_of 取得正確的 MIME。
"""
import io
import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import List, Optional, Tuple

from gtts import gTTS

ONLINE_TIMEOUT   = 3.0     # gTTS 連線逾時（秒）；展場網路掛掉時最多等這麼久就改離線
ONLINE_RETRY_SEC = 60.0    # 線上失敗後，這段時間內直接走離線
OFFLINE_TIMEOUT  = 20.0    # say 合成逾時（正常不到 1 秒）
DEFAULT_VOICE    = "Meijia"

_lock = threading.Lock()
_online_down_until = 0.0
_voice_cache: Optional[str] = None
_voice_checked = False


def _mode() -> str:
    m = os.getenv("TTS_BACKEND", "auto").strip().lower()
    return m if m in ("auto", "online", "offline") else "auto"


# ── 本機語音（macOS say）──────────────────────────
def _list_voices() -> List[Tuple[str, str]]:
    """`say -v '?'` → [(語音名稱, 語系), ...]；沒有 say 或失敗回傳空串列。"""
    if platform.system() != "Darwin" or not shutil.which("say"):
        return []
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True,
                             timeout=10).stdout
    except Exception:
        return []
    voices = []
    for line in out.splitlines():
        # 例：「Meijia              zh_TW    # 你好，我叫美佳。」
        m = re.match(r"^(.+?)\s+([a-z]{2}_[A-Z]{2})\s+#", line)
        if m:
            voices.append((m.group(1).strip(), m.group(2)))
    return voices


def _pick_voice() -> Optional[str]:
    """挑本機中文語音：TTS_VOICE > Meijia > 其他台灣中文 > 其他中文；都沒有回傳 None。"""
    global _voice_cache, _voice_checked
    if _voice_checked:
        return _voice_cache
    with _lock:
        if _voice_checked:
            return _voice_cache
        voices = _list_voices()
        wanted = os.getenv("TTS_VOICE", "").strip()
        names = {n for n, _ in voices}
        pick = None
        if wanted and wanted in names:
            pick = wanted
        elif wanted:
            print(f"[TTS] 找不到本機語音「{wanted}」，改自動挑選")
        if pick is None:
            tw = [n for n, loc in voices if loc == "zh_TW"]
            zh = [n for n, loc in voices if loc.startswith("zh_")]
            pick = (DEFAULT_VOICE if DEFAULT_VOICE in tw else None) or (tw[0] if tw else None) \
                or (zh[0] if zh else None)
        _voice_cache, _voice_checked = pick, True
        return pick


def offline_available() -> bool:
    return _pick_voice() is not None


def _offline(text: str) -> Optional[bytes]:
    """用 macOS say 合成 wav；文字走 stdin（避免以 - 開頭或太長的參數問題）。"""
    voice = _pick_voice()
    if voice is None:
        print("[TTS] 沒有可用的本機中文語音（目前離線朗讀只支援 macOS）")
        return None
    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        r = subprocess.run(
            ["say", "-v", voice, "-f", "-", "-o", path,
             "--file-format=WAVE", "--data-format=LEI16@22050"],
            input=text.encode("utf-8"), capture_output=True, timeout=OFFLINE_TIMEOUT)
        if r.returncode != 0:
            print(f"[TTS 本機語音錯誤] {r.stderr.decode('utf-8', 'replace').strip()}")
            return None
        with open(path, "rb") as f:
            data = f.read()
        return data or None
    except Exception as e:
        print(f"[TTS 本機語音錯誤] {e}")
        return None
    finally:
        if path and os.path.exists(path):
            os.remove(path)


# ── 線上（gTTS）──────────────────────────────────
def _online(text: str, lang: str) -> Optional[bytes]:
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang=lang, slow=False, timeout=ONLINE_TIMEOUT).write_to_fp(buf)
        return buf.getvalue() or None
    except Exception as e:
        print(f"[TTS 線上錯誤] {type(e).__name__}: {str(e)[:120]}")
        return None


# ── 對外介面 ─────────────────────────────────────
def synthesize(text: str, lang: str = "zh-TW") -> Optional[bytes]:
    """將文字合成為音訊位元組（線上 mp3 或離線 wav，用 mime_of 判斷），失敗回傳 None。"""
    global _online_down_until
    if not text:
        return None
    mode = _mode()

    if mode != "offline" and time.monotonic() >= _online_down_until:
        data = _online(text, lang)
        if data:
            return data
        if mode == "online":
            return None
        with _lock:
            _online_down_until = time.monotonic() + ONLINE_RETRY_SEC
        voice = _pick_voice()
        print(f"[TTS] 連不上 Google 語音，改用本機語音"
              f"{'「' + voice + '」' if voice else '（沒有可用的）'}；"
              f"{int(ONLINE_RETRY_SEC)} 秒後再試線上")

    return _offline(text)


def mime_of(data: bytes) -> str:
    """音訊位元組的 MIME：本機語音是 wav（RIFF 開頭），其餘視為 gTTS 的 mp3。"""
    return "audio/wav" if data[:4] == b"RIFF" else "audio/mpeg"


def backend_status() -> str:
    """給網頁標題列的一句話：現在朗讀用什麼。"""
    voice = _pick_voice()
    mode = _mode()
    if mode == "offline":
        return f"本機語音 {voice}（離線）" if voice else "本機語音（找不到可用的中文語音）"
    if mode == "online":
        return "gTTS（需要網路）"
    return (f"gTTS，連不上網路時自動改用本機語音 {voice}" if voice
            else "gTTS（需要網路；本機沒有可用的離線語音）")
