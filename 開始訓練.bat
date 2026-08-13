@echo off
REM Windows 專用：雙擊這個檔案就會開始訓練。
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ================================================
echo   手語辨識模型　一鍵訓練 (Windows)
echo ================================================

set "PY="

REM 第一輪：找「套件已經裝好」的 Python，直接可以訓練
for %%C in ("py -3" "python" "python3") do (
    if not defined PY (
        %%~C -c "import numpy, torch, sklearn" >nul 2>&1 && (
            set "PY=%%~C"
            echo 找到已裝好套件的 Python：%%~C
        )
    )
)

REM 第二輪：沒有現成的就挑任何可用的，讓 start_training.py 去安裝套件
for %%C in ("py -3" "python" "python3") do (
    if not defined PY (
        %%~C --version >nul 2>&1 && (
            set "PY=%%~C"
            echo 使用 Python：%%~C  ^(等一下會自動安裝需要的套件^)
        )
    )
)

if not defined PY (
    echo.
    echo [X] 找不到 Python。
    echo.
    echo  ^> 請先安裝 Python 3.10：https://www.python.org/downloads/
    echo    安裝時務必勾選 "Add Python to PATH"，裝完重新雙擊這個檔案。
    echo.
    pause
    exit /b 1
)

echo 版本：
%PY% --version
echo.
%PY% start_training.py

if errorlevel 1 pause
endlocal
