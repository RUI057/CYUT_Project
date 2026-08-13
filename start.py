#!/usr/bin/env python3
"""一鍵操作選單：收集資料、訓練模型、測試辨識。

給第一次接觸這個專案的人，開起來照著選單按就好。
會自動檢查並安裝各功能需要的套件。

使用方式：
  - 雙擊「開始.command」(Mac) 或「開始.bat」(Windows)
  - 或終端機執行： python start.py
"""
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

BAR = "═" * 58
IS_TTY = sys.stdin is not None and sys.stdin.isatty()

# 各功能需要的套件：import 名稱 → pip 安裝名稱
PKG_COLLECT = [("cv2", "opencv-python>=4.9.0"), ("mediapipe", "mediapipe>=0.10.7,<0.11"),
               ("PIL", "Pillow>=10.0.0"), ("numpy", "numpy>=1.26,<2.0")]
PKG_TRAIN   = [("numpy", "numpy>=1.26,<2.0"), ("torch", "torch>=2.2.0"),
               ("sklearn", "scikit-learn>=1.4.0")]


def title(msg):
    print(f"\n{BAR}\n  {msg}\n{BAR}")


def pause():
    if IS_TTY:
        try:
            input("\n按 Enter 回到選單...")
        except (EOFError, KeyboardInterrupt):
            pass


def ask_yes(msg):
    """詢問是否繼續；直接按 Enter 視為是，讀不到輸入（EOF）視為否。"""
    try:
        return (input(f"{msg} [Y/n] ").strip().lower() or "y") in ("y", "yes", "是")
    except (EOFError, KeyboardInterrupt):
        return False


def ensure_packages(pkgs, what):
    """檢查套件，缺的話問要不要裝。回傳 True 表示可以繼續。"""
    missing = []
    for mod, spec in pkgs:
        try:
            __import__(mod)
        except ImportError:
            missing.append((mod, spec))
    if not missing:
        return True

    print(f"\n「{what}」需要這些套件，但還沒安裝：")
    for mod, spec in missing:
        print(f"  - {spec}")
    if not ask_yes("\n要現在自動安裝嗎？"):
        print("\n已取消。你也可以手動安裝：")
        print(f"  {sys.executable} -m pip install " + " ".join(s for _, s in missing))
        return False

    cmd = [sys.executable, "-m", "pip", "install"] + [s for _, s in missing]
    print(f"\n執行：{' '.join(cmd)}\n")
    if subprocess.call(cmd) != 0:
        print("\n第一次安裝失敗，改用 --user 再試一次...\n")
        if subprocess.call(cmd + ["--user"]) != 0:
            print("\n❌ 安裝失敗。多半是系統 Python 被保護，建議改用虛擬環境：")
            print(f"     {sys.executable} -m venv .venv")
            print("     source .venv/bin/activate   (Windows: .venv\\Scripts\\activate)")
            print("     pip install -r requirements.txt")
            return False

    for mod, _ in missing:
        try:
            __import__(mod)
        except ImportError:
            print(f"\n❌ 安裝完仍找不到 {mod}，可能裝到別的 Python 環境了。")
            return False
    print("\n套件安裝完成 ✅")
    return True


def data_counts():
    """回傳 {詞: 筆數}（只算格式正確的）。"""
    import numpy as np
    from src.vocab import get_all_labels
    from src.features import HANDS_DIM, FEAT_DIM

    counts = Counter()
    for label in get_all_labels():
        folder = ROOT / "dynamic_dataset" / label
        if not folder.exists():
            continue
        for f in folder.glob("*.npy"):
            try:
                shape = np.load(f, mmap_mode="r").shape
            except Exception:
                continue
            if len(shape) == 2 and shape[0] == 30 and shape[1] in (HANDS_DIM, FEAT_DIM):
                counts[label] += 1
    return counts


def show_progress():
    """顯示收集進度：哪些詞收了、哪些還沒。"""
    from src.vocab import get_all_labels
    labels = get_all_labels()
    counts = data_counts()
    done   = [l for l in labels if counts.get(l, 0) > 0]
    todo   = [l for l in labels if counts.get(l, 0) == 0]
    total  = sum(counts.values())

    title("資料收集進度")
    print(f"總樣本數：{total} 筆")
    print(f"已開始收集：{len(done)} / {len(labels)} 個詞")

    if done:
        print("\n【已收集】（筆數少的排前面，優先補）")
        for l in sorted(done, key=lambda x: counts[x]):
            mark = "  ← 偏少" if counts[l] < 30 else ""
            print(f"  {l:<8}{counts[l]:>5}{mark}")
    if todo:
        print(f"\n【還沒收集】共 {len(todo)} 個")
        for i in range(0, min(len(todo), 60), 10):
            print("  " + "、".join(todo[i:i + 10]))
        if len(todo) > 60:
            print(f"  ...（其餘 {len(todo) - 60} 個）")
    return labels, counts, todo


def do_collect():
    if not ensure_packages(PKG_COLLECT, "收集資料"):
        return
    labels, counts, todo = show_progress()

    print("\n" + "─" * 58)
    print("要從哪個詞開始收集？")
    print("  直接按 Enter = 第一個還沒收滿的詞")
    print("  或輸入「詞名」，例如：開心")
    print("  或輸入編號（上面清單的順序）")
    try:
        ans = input("\n要收集：").strip()
    except (EOFError, KeyboardInterrupt):
        return

    cmd = [sys.executable, "collect_data.py"]
    if ans:
        if ans.isdigit() and 1 <= int(ans) <= len(labels):
            cmd += ["--word", labels[int(ans) - 1]]
        elif ans in labels:
            cmd += ["--word", ans]
        else:
            print(f"\n❌ 詞彙表裡沒有「{ans}」")
            return

    title("即將開啟攝影機")
    print("操作說明：")
    print("  手放進畫面 → 倒數 3-2-1 → 比出手勢 → 自動存檔")
    print("  N=下一個詞    M=上一個詞    J=跳到指定詞")
    print("  D=刪除上一筆  R=重置此詞    S=暫停    Q=離開")
    print("\n⚠️ 第一次執行時，系統會詢問攝影機權限，請按「允許」。")
    print("   若沒有出現畫面，檢查有沒有別的程式正在用攝影機。")
    if not ask_yes("\n準備好了嗎？"):
        return
    subprocess.call(cmd)


def do_train():
    if not ensure_packages(PKG_TRAIN, "訓練模型"):
        return
    counts = data_counts()
    if not counts:
        print("\n❌ 還沒有任何訓練資料，請先選 1 收集資料。")
        return
    if len(counts) < 2:
        print(f"\n❌ 只有 1 個詞有資料（{list(counts)[0]}），至少要 2 個詞才能訓練分類模型。")
        return

    total = sum(counts.values())
    title("訓練模型")
    print(f"將用 {total} 筆資料、{len(counts)} 個詞來訓練。")
    print("\n選擇模式：")
    print("  1. 完整訓練（含交叉驗證，會算出準確率，比較久）")
    print("  2. 快速訓練（跳過驗證，只求把模型練出來）")
    try:
        mode = input("\n請選擇 [1]：").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        return

    if mode == "2":
        # 跳過交叉驗證：直接用全部資料訓練並存檔
        code = (
            "import pickle, numpy as np;"
            "from src.trainer import load_dataset, train_full, DEVICE, SEQ_LEN;"
            "from src.features import FEAT_DIM;"
            "from src.model import GestureLSTM;"
            "X, y, g, c = load_dataset();"
            "f = lambda: GestureLSTM(feat_dim=FEAT_DIM, num_classes=len(c)).to(DEVICE);"
            "s = train_full(f, X, y, len(c));"
            "pickle.dump({'model_state': s, 'classes': c, 'seq_len': SEQ_LEN,"
            " 'feat_dim': FEAT_DIM, 'hidden': 128, 'num_layers': 2,"
            " 'cv_acc_mean': 0.0, 'cv_acc_std': 0.0}, open('data/model.pkl','wb'));"
            "print('\\n[完成] 模型已存到 data/model.pkl')"
        )
        rc = subprocess.call([sys.executable, "-c", code])
    else:
        print("\n開始完整訓練，請耐心等待（有 GPU 約 15 分鐘，純 CPU 可能 1 小時以上）...\n")
        rc = subprocess.call([sys.executable, "train_model.py"])

    if rc == 0:
        print("\n✅ 訓練完成，模型存在 data/model.pkl")
    else:
        print("\n❌ 訓練沒有正常完成，請把上面的錯誤訊息傳給組長。")


def do_test():
    if not ensure_packages(PKG_COLLECT + PKG_TRAIN, "測試辨識"):
        return
    if not (ROOT / "data" / "model.pkl").exists():
        print("\n❌ 找不到 data/model.pkl，請先選 2 訓練模型。")
        return
    title("即將開啟攝影機測試辨識")
    print("比出手勢，畫面上會顯示辨識結果。按 Q 離開。")
    if not ask_yes("\n準備好了嗎？"):
        return
    subprocess.call([sys.executable, "test_model.py"])


def do_check():
    if not ensure_packages([("numpy", "numpy>=1.26,<2.0")], "檢查資料品質"):
        return
    subprocess.call([sys.executable, "check_dataset.py"])


MENU = """
  1. 收集資料      用攝影機錄製手語動作
  2. 訓練模型      用收集到的資料訓練
  3. 測試辨識      用攝影機測試訓練好的模型
  4. 查看進度      看哪些詞收了、哪些還沒
  5. 檢查資料品質  找出錄壞的樣本
  0. 離開
"""


def main():
    v = sys.version_info
    if v < (3, 9):
        print(f"❌ Python 版本太舊（{v.major}.{v.minor}），需要 3.9 以上。")
        print("👉 到 https://www.python.org/downloads/ 安裝 Python 3.10 或 3.11。")
        pause()
        return

    actions = {"1": do_collect, "2": do_train, "3": do_test,
               "4": lambda: show_progress(), "5": do_check}

    while True:
        title("手語辨識專案　操作選單")
        print(f"Python {v.major}.{v.minor}.{v.micro}　工作目錄：{ROOT.name}")
        print(MENU)
        try:
            choice = input("請輸入編號：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if choice in ("0", "q", "Q"):
            print("\n掰掰！")
            return
        act = actions.get(choice)
        if not act:
            print(f"\n沒有「{choice}」這個選項，請重新輸入。")
            continue
        try:
            act()
        except KeyboardInterrupt:
            print("\n\n已中斷。")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n❌ 發生錯誤：{type(e).__name__}: {e}")
            print("   請把完整錯誤訊息傳給組長。")
        pause()


if __name__ == "__main__":
    main()
