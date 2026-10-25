#!/usr/bin/env python3
"""產生「手語收集包」：給只負責收集資料的組員，不含訓練、測試與既有資料。

分工：組員用 Windows 只收集（收集包）；組長用 Mac 收集＋訓練（主專案，雙擊開始.command）。
所以收集包只放 Windows 需要的東西。

用法：python make_collect_kit.py
產出：手語收集包/（資料夾）與 手語收集包.zip（直接傳給組員）

收集包裡的程式都是「從主專案複製」而不是另外維護一份：特徵抽取（src/features.py）
只要跟主專案不一致，組員收到的資料就和你的對不上。所以主專案的收集程式、
features.py、詞彙表或比法提示改過之後，**要重新執行這支程式再發給組員**。

Windows 的「開始.bat」不放在主專案，而是由本程式的 WIN_LAUNCHER 範本寫進收集包
（要改 Windows 啟動流程就改那段範本）。

收集包的 start.py / setup_env.py 會偵測到沒有 train_model.py，自動變成純收集模式：
選單只有「收集／查看進度／打包資料」，也不安裝訓練用的 torch、scikit-learn。
"""
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "手語收集包"
OUT  = ROOT / NAME
ZIP  = ROOT / f"{NAME}.zip"

COPY = [
    "setup_env.py",                      # Windows 自動建環境
    "start.py",                          # 選單（偵測到沒有 train_model.py → 純收集模式）
    "collect_data.py",
    "src/features.py", "src/vocab.py", "src/video.py", "src/font_utils.py",
    "data/vocabulary.json", "data/hints.json",
]
# 放在收集包最外層、給組員看的參考檔（程式不會讀）：主專案路徑 → 收集包裡的檔名
EXTRA = {
    "data/手語演示網址.txt": "手語演示網址.txt",   # 比法提示大多是空的，看示範影片才知道怎麼比
}
# Windows 雙擊啟動器：找相容的 Python（3.9～3.12）→ 執行 setup_env.py；沒有 Python 時可用 winget 自動安裝。
# 寫入時轉成 CRLF、不加 BOM：cmd 遇到 LF 行尾可能找不到 goto 標籤。
WIN_LAUNCHER = r'''@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title 手語辨識專案
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

rem ==============================================================
rem  Windows 專用：雙擊就會開啟操作選單（收集資料 / 訓練 / 測試）。
rem  第一次會自動建立專用環境、安裝套件（約 5 到 20 分鐘），之後幾秒就開好。
rem  設定流程都在 setup_env.py；這裡只負責找到版本相容的 Python。
rem ==============================================================

set "PY="

rem 1. 已經設定好的專用環境：直接用
if exist "%PUBLIC%\cyut_venv\Scripts\python.exe" set PY="%PUBLIC%\cyut_venv\Scripts\python.exe"
if defined PY %PY% -c "import sys" >nul 2>&1 || set "PY="
if defined PY goto run

rem 2. 找相容的 Python：3.9 到 3.12（3.13 以上 mediapipe 還不支援）
for %%V in (3.11 3.12 3.10 3.9) do if not defined PY py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
if defined PY goto run
for %%C in (python python3) do if not defined PY %%C -c "import sys; sys.exit(0 if (3,9) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1 && set "PY=%%C"
if defined PY goto run
goto nopython

:run
%PY% setup_env.py
if errorlevel 1 pause
goto end

:nopython
echo.
echo  [X] 找不到可用的 Python（需要 3.9 到 3.12 版）
echo.
echo  目前電腦上裝的 Python：
py -0 2>nul || echo    （沒有偵測到）
echo.
echo  注意：Python 3.13 以上太新，mediapipe 還不支援，請安裝 3.11。
echo.
where winget >nul 2>&1 || goto manual
choice /C YN /M "要現在自動安裝 Python 3.11 嗎"
if errorlevel 2 goto manual
winget install -e --id Python.Python.3.11
echo.
echo  安裝完成後，請關閉這個視窗，再雙擊一次「開始.bat」。
pause
goto end

:manual
echo  請下載 Python 3.11（頁面往下捲，選 Windows installer 64-bit）：
echo    https://www.python.org/downloads/release/python-3119/
echo  安裝時記得勾選「Add python.exe to PATH」，裝完再雙擊一次「開始.bat」。
start "" "https://www.python.org/downloads/release/python-3119/"
pause
goto end

:end
endlocal
'''

TRAIN_ONLY = {"torch", "scikit-learn"}   # 收集用不到，省下約 140 MB 下載

GUIDE = """\
# 手語資料收集包

只負責「收集資料」的精簡版。收完打包傳給組長就好，不需要訓練。

## 1. 開起來

雙擊 **`開始.bat`** 就會開啟操作選單。

- 需要 **Python 3.9～3.12（建議 3.11）**，**不要裝 3.13 以上**（太新，套件還不支援）。
  Windows 沒有 Python 時，`開始.bat` 會問你要不要自動安裝；
  也可以自己下載：https://www.python.org/downloads/release/python-3119/
  （選 *Windows installer 64-bit*，安裝時勾選「**Add python.exe to PATH**」）
- Windows 第一次會自動設定環境：下載約 280 MB、**3～10 分鐘**，畫面一直跑字是正常的。
  之後再開，幾秒就會進選單。

## 2. 收集資料

選單選 **1**，輸入要收的詞（直接按 Enter 就從第一個還沒收滿的詞開始），攝影機會打開。

**流程**：手放進畫面 → 倒數 3-2-1 → 比出手勢 → 自動存檔 → 換個角度再比一次

| 鍵 | 功能 |
|---|---|
| `N` / `M` | 下一個 / 上一個詞 |
| `J` | 跳到指定詞（回黑色文字視窗輸入詞名） |
| `D` | 刪除剛剛那筆（比錯了就按） |
| `S` | 暫停 / 繼續 |
| `Q` | 離開 |

**不知道怎麼比？** 打開收集包裡的 **`手語演示網址.txt`**，找到那個詞點影片看示範
（畫面上的比法提示只有少數詞有）。

**要點**：
- **手要完整入鏡**：畫面左上會顯示「偵測到 N 隻手」，雙手的詞兩隻手都要在畫面裡、不要被遮住
- 看到橘字「**注意：上一筆手連續 N 幀沒抓到**」→ 動作太快或光線不足，按 `D` 刪掉重錄，
  放慢一點或把燈打亮；右上角 `FPS` 變紅字也是光線太暗的訊號
- 每筆之間**換個距離或角度**，資料多樣模型才準
- 比錯了立刻按 `D` 刪掉

## 3. 交給組長

選單選 **3 打包資料** → 輸入你的名字 → 會產生 `手語資料_名字_日期.zip`，並自動打開它所在的資料夾。

**把這個 zip 傳給組長就好，不要自己另外壓縮**（自己壓縮的話，中文資料夾名可能變成亂碼）。
原本的資料會留著，之後可以繼續收、再打包一次；組長合併時會自動略過重複的。

## 常見狀況

- **Windows 跳出「Windows 已保護您的電腦」**：按「其他資訊」→「仍要執行」。
- **攝影機打不開／黑畫面**：先關掉其他用到攝影機的程式（視訊會議、LINE、瀏覽器）。
  - 設定 → 隱私權與安全性 → 相機 → 開啟「讓桌面應用程式存取您的相機」
- **按鍵沒反應**：先用滑鼠點一下攝影機畫面，讓它在最上層。
- **Windows 出現「缺少 Visual C++」或「DLL load failed」**：
  安裝 https://aka.ms/vs/17/release/vc_redist.x64.exe 後重新開啟。
- **Windows 環境壞掉想重來**：刪除 `C:\\Users\\Public\\cyut_venv` 資料夾，再雙擊 `開始.bat`。
- **出現紅字錯誤**：把**完整**畫面截圖傳給組長。
"""


def fail(msg):
    print(f"❌ {msg}")
    sys.exit(1)


def git_version():
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *COPY, *EXTRA, "requirements-core.txt"],
                               cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return f"commit {h}" + ("（含未提交變更）" if dirty else "") if h else "非 git 版本"
    except OSError:
        return "非 git 版本"


def kit_requirements():
    """主專案的 requirements-core.txt 拿掉訓練用套件，版本限制完全沿用。"""
    lines = ["# 純收集版：由 make_collect_kit.py 從主專案產生，請勿手動修改。",
             "# 拿掉了訓練用的 torch、scikit-learn；其餘版本限制與原因同主專案。", ""]
    for line in (ROOT / "requirements-core.txt").read_text(encoding="utf-8").splitlines():
        pkg = re.split(r"[<>=!~;\[\s]", line.split("#", 1)[0].strip(), 1)[0].lower()
        if pkg not in TRAIN_ONLY:
            lines.append(line)
    return "\n".join(lines) + "\n"


def check_self_contained(out):
    """收集包裡的程式 import 到的 src 模組都要在包裡，否則組員一開就壞。"""
    missing = set()
    for py in out.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for mod in re.findall(r"^\s*from src\.(\w+) import", text, re.M) + \
                   re.findall(r"^\s*from src import (\w+)", text, re.M):
            if not (out / "src" / f"{mod}.py").exists():
                missing.add(f"{py.relative_to(out)} → src/{mod}.py")
    return sorted(missing)


def main():
    for f in COPY + list(EXTRA) + ["requirements-core.txt"]:
        if not (ROOT / f).exists():
            fail(f"主專案缺少 {f}")

    if OUT.exists():
        if not (OUT / "KIT_VERSION.txt").exists():
            fail(f"{OUT.name}/ 已存在但不是這支程式產生的，為了安全不會刪除。請先手動移走。")
        collected = list(OUT.glob("dynamic_dataset/*/*.npy"))
        if collected:
            fail(f"{OUT.name}/ 裡有 {len(collected)} 筆收集到的資料，重建會刪掉它們。\n"
                 f"   請先在收集包選單用「打包資料」匯出，再用 merge_data.py 合併回主專案。")
        shutil.rmtree(OUT)

    for rel in COPY:
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    for rel, name in EXTRA.items():
        shutil.copy2(ROOT / rel, OUT / name)

    # 行尾一定要是 CRLF（否則 cmd 可能找不到 goto 標籤）、不加 BOM
    (OUT / "開始.bat").write_bytes(WIN_LAUNCHER.replace("\n", "\r\n").encode("utf-8"))

    (OUT / "requirements-core.txt").write_text(kit_requirements(), encoding="utf-8")
    (OUT / "使用說明.md").write_text(GUIDE, encoding="utf-8")

    sys.path.insert(0, str(ROOT))
    from src.vocab import get_all_labels
    n_words = len(get_all_labels())
    built = datetime.now().strftime("%Y-%m-%d %H:%M")
    (OUT / "KIT_VERSION.txt").write_text(
        f"手語收集包 {built} 建立（主專案 {git_version()}）\n詞彙：{n_words} 個\n", encoding="utf-8")

    missing = check_self_contained(OUT)
    if missing:
        fail("收集包缺少模組（請把它加進 make_collect_kit.py 的 COPY）：\n   " + "\n   ".join(missing))

    # 用 Python 打包：檔名標 UTF-8（Windows 解壓縮時中文資料夾名不亂碼）
    ZIP.unlink(missing_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                z.write(f, f"{NAME}/{f.relative_to(OUT).as_posix()}")

    files = sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file())
    print(f"✅ 已產生 {OUT.name}/（{len(files)} 個檔案）與 {ZIP.name}（{ZIP.stat().st_size / 1e3:.0f} KB）")
    for f in files:
        print(f"   {f}")
    print(f"\n版本：{(OUT / 'KIT_VERSION.txt').read_text(encoding='utf-8').splitlines()[0]}")
    print(f"把 {ZIP.name} 傳給組員即可。主專案的收集程式、詞彙表或提示改過後，記得重新執行本程式。")


if __name__ == "__main__":
    main()
