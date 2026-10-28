#!/bin/bash
# Mac 專用：雙擊直接開啟手語翻譯網頁（不經過選單），瀏覽器會自動打開。
# 若雙擊沒反應，先在終端機執行一次：chmod +x "翻譯.command"
# 找 Python、檢查套件的邏輯都在「開始.command」，這裡只是帶 --web 呼叫它。
cd "$(dirname "$0")" || exit 1
exec /bin/bash ./開始.command --web
