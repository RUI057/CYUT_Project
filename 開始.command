#!/bin/bash
# Mac 專用：雙擊這個檔案就會開啟操作選單（收集資料 / 訓練模型 / 測試辨識）。
# 若雙擊沒反應，先在終端機執行一次：chmod +x "開始.command"

cd "$(dirname "$0")" || exit 1

echo "================================================"
echo "  手語辨識專案 (Mac)"
echo "================================================"

has_pkgs() { "$1" -c "import numpy, cv2, mediapipe" >/dev/null 2>&1; }

# 候選 Python：conda 環境 → 具體版本 → 通用名稱
CANDIDATES=()
[ -n "$CONDA_PREFIX" ] && CANDIDATES+=("$CONDA_PREFIX/bin/python")
for p in /opt/*conda3/envs/*/bin/python "$HOME"/*conda3/envs/*/bin/python \
         /opt/*conda3/bin/python "$HOME"/*conda3/bin/python; do
    [ -x "$p" ] && CANDIDATES+=("$p")
done
for name in python3.11 python3.10 python3.12 python3 python; do
    p=$(command -v "$name" 2>/dev/null) && CANDIDATES+=("$p")
done

# 第一輪：挑「套件已經裝好」的，開起來就能用
PY=""
for c in "${CANDIDATES[@]}"; do
    if has_pkgs "$c"; then PY="$c"; echo "找到已裝好套件的 Python：$c"; break; fi
done

# 第二輪：沒有現成的就挑任何可用的，讓選單去安裝套件
if [ -z "$PY" ]; then
    for c in "${CANDIDATES[@]}"; do
        if "$c" -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)" 2>/dev/null; then
            PY="$c"; echo "使用 Python：$c（需要時會自動安裝套件）"; break
        fi
    done
fi

if [ -z "$PY" ]; then
    echo ""
    echo "❌ 找不到可用的 Python 3.9 以上版本。"
    echo ""
    echo "👉 請先安裝 Python 3.10：https://www.python.org/downloads/"
    echo "   裝完重新雙擊這個檔案。"
    echo ""
    read -r -p "按 Enter 關閉視窗..."
    exit 1
fi

echo "版本：$("$PY" --version 2>&1)"
"$PY" start.py
