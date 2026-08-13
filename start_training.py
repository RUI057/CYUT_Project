#!/usr/bin/env python3
"""一鍵訓練：給第一次接觸這個專案的人，開起來就能訓練手語模型。

會自動做完這些事：
  1. 檢查 Python 版本
  2. 檢查並安裝缺少的套件（只需 numpy / torch / scikit-learn，不需 mediapipe）
  3. 檢查資料，列出每個詞的樣本數
  4. 訓練模型並存成 data/model.pkl

使用方式（三選一）：
  - 直接雙擊「開始訓練.command」(Mac) 或「開始訓練.bat」(Windows)
  - 終端機執行： python start_training.py
  - 進階：python start_training.py --quick --yes --epochs 30
"""
import argparse
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# 不論從哪裡執行，都切到專案根目錄（雙擊執行時 cwd 常常是家目錄）
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

BAR = "─" * 60
IS_TTY = sys.stdin is not None and sys.stdin.isatty()


def title(msg):
    print(f"\n{BAR}\n  {msg}\n{BAR}")


def pause_exit(code=0):
    """雙擊執行時，視窗不要一閃就消失。"""
    if IS_TTY:
        try:
            input("\n按 Enter 關閉視窗...")
        except (EOFError, KeyboardInterrupt):
            pass
    sys.exit(code)


def die(msg, hint=""):
    print(f"\n❌ {msg}")
    if hint:
        print(f"\n👉 怎麼解決：{hint}")
    pause_exit(1)


def ask(msg, default_yes=True, auto_yes=False):
    if auto_yes or not IS_TTY:
        return True
    suffix = "[Y/n]" if default_yes else "[y/N]"
    try:
        ans = input(f"{msg} {suffix} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if not ans:
        return default_yes
    return ans in ("y", "yes", "是")


# ── 1. Python 版本 ────────────────────────────────────────
def check_python():
    v = sys.version_info
    print(f"Python 版本：{v.major}.{v.minor}.{v.micro}")
    if v < (3, 9):
        die(f"Python 版本太舊（{v.major}.{v.minor}），需要 3.9 以上。",
            "到 https://www.python.org/downloads/ 安裝 Python 3.10 或 3.11，再重新執行。")
    if v >= (3, 13):
        print("⚠️  Python 3.13 以上較新，部分套件可能還沒有對應版本；"
              "若安裝失敗請改用 3.10～3.12。")


# ── 2. 套件 ──────────────────────────────────────────────
# 訓練只需要這三個；mediapipe / streamlit 是「收集資料」和「網頁」才需要
NEEDED = [("numpy", "numpy>=1.26,<2.0"),
          ("torch", "torch>=2.2.0"),
          ("sklearn", "scikit-learn>=1.4.0")]


def check_packages(auto_yes=False):
    missing = []
    for mod, spec in NEEDED:
        try:
            __import__(mod)
        except ImportError:
            missing.append((mod, spec))

    if not missing:
        import numpy, torch, sklearn
        print(f"套件檢查：OK（numpy {numpy.__version__}, "
              f"torch {torch.__version__}, scikit-learn {sklearn.__version__}）")
        return

    names = "、".join(m for m, _ in missing)
    print(f"缺少套件：{names}")
    if not ask(f"要現在自動安裝嗎？（會執行 pip install）", auto_yes=auto_yes):
        die("缺少必要套件，無法訓練。",
            f"手動安裝：{sys.executable} -m pip install " + " ".join(s for _, s in missing))

    specs = [s for _, s in missing]
    cmd = [sys.executable, "-m", "pip", "install"] + specs
    print(f"\n執行：{' '.join(cmd)}\n")
    if subprocess.call(cmd) != 0:
        # macOS/Linux 新版常見：系統 Python 被保護（externally-managed-environment）
        print("\n第一次安裝失敗，改用 --user 再試一次...\n")
        if subprocess.call(cmd + ["--user"]) != 0:
            die("套件安裝失敗。",
                "多半是系統 Python 被保護。建議改用虛擬環境：\n"
                f"     {sys.executable} -m venv .venv\n"
                "     source .venv/bin/activate     (Windows: .venv\\Scripts\\activate)\n"
                "     pip install -r requirements.txt\n"
                "     python start_training.py")

    for mod, _ in missing:
        try:
            __import__(mod)
        except ImportError:
            die(f"安裝完仍然找不到 {mod}。",
                "可能裝到別的 Python 環境了；請確認用同一個 python 執行本程式。")
    print("\n套件安裝完成 ✅")


# ── 3. 資料 ──────────────────────────────────────────────
def check_data():
    from src.vocab import get_all_labels
    from src.features import HANDS_DIM, FEAT_DIM
    import numpy as np

    data_dir = ROOT / "dynamic_dataset"
    if not data_dir.exists():
        die("找不到 dynamic_dataset 資料夾。",
            "請確認你是完整 clone 這個 git 專案（資料有一起放在 repo 裡）。")

    counts, bad = Counter(), 0
    for label in get_all_labels():
        folder = data_dir / label
        if not folder.exists():
            continue
        for f in folder.glob("*.npy"):
            try:
                shape = np.load(f, mmap_mode="r").shape
            except Exception:
                bad += 1
                continue
            if len(shape) == 2 and shape[0] == 30 and shape[1] in (HANDS_DIM, FEAT_DIM):
                counts[label] += 1
            else:
                bad += 1

    if not counts:
        die("dynamic_dataset 裡沒有任何可用的訓練資料。",
            "請先用 collect_data.py 收集資料，或向組長拿完整的 dynamic_dataset 資料夾。")

    total = sum(counts.values())
    print(f"可用資料：{total} 筆，涵蓋 {len(counts)} 個詞")
    if bad:
        print(f"⚠️  有 {bad} 個檔案格式不符，會自動跳過")

    print("\n各詞樣本數：")
    for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        mark = "  ← 偏少" if n < 30 else ""
        print(f"  {label:<8}{n:>5}{mark}")

    weak = [l for l, n in counts.items() if n < 30]
    if weak:
        print(f"\n⚠️  這些詞樣本偏少（<30），準確率會比較差：{'、'.join(weak)}")
    if len(counts) < 2:
        die("只有 1 個詞有資料，無法訓練分類模型（至少要 2 個詞）。",
            "請先收集更多不同詞彙的資料。")
    return total, len(counts)


# ── 4. 訓練 ──────────────────────────────────────────────
def run_training(quick, epochs, folds, out_path=None):
    import numpy as np
    from src.trainer import (load_dataset, cross_validate, train_full,
                             confusion_pairs, full_report, DEVICE, SEQ_LEN)
    from src.features import FEAT_DIM
    from src.model import GestureLSTM
    import pickle

    print(f"\n運算裝置：{DEVICE}"
          + ("（GPU 加速）" if str(DEVICE) != "cpu" else "（CPU，速度較慢是正常的）"))

    X, y, groups, classes = load_dataset()
    n_class = len(classes)
    factory = lambda: GestureLSTM(feat_dim=FEAT_DIM, num_classes=n_class).to(DEVICE)
    n_param = sum(p.numel() for p in factory().parameters())
    print(f"模型：Bi-LSTM  參數量 {n_param:,}  特徵維度 {FEAT_DIM}")

    t0 = time.time()
    cv_mean = cv_std = 0.0

    if quick:
        print(f"\n快速模式：跳過交叉驗證，直接用全部資料訓練 {epochs} epoch")
    else:
        title(f"步驟 1/2　交叉驗證（{folds}-fold，用來估算真實準確率）")
        fold_accs, agg_true, agg_pred = cross_validate(
            factory, X, y, groups, n_class, n_folds=folds, epochs=epochs)
        cv_mean, cv_std = float(np.mean(fold_accs)), float(np.std(fold_accs))
        print(f"\n交叉驗證準確率：{cv_mean:.1%} ± {cv_std:.1%}")
        print("\n[分類報告]")
        full_report(agg_true, agg_pred, classes)
        confusion_pairs(agg_true, agg_pred, classes)
        title("步驟 2/2　用全部資料訓練最終模型")

    state = train_full(factory, X, y, n_class, epochs=epochs)

    out = Path(out_path) if out_path else ROOT / "data" / "model.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"model_state": state, "classes": classes, "seq_len": SEQ_LEN,
                     "feat_dim": FEAT_DIM, "hidden": 128, "num_layers": 2,
                     "cv_acc_mean": cv_mean, "cv_acc_std": cv_std}, f)

    mins = (time.time() - t0) / 60
    title("訓練完成 ✅")
    print(f"模型已存到：{out}")
    print(f"詞彙數：{n_class}　訓練資料：{len(X)} 筆　耗時：{mins:.1f} 分鐘")
    if not quick:
        print(f"交叉驗證準確率：{cv_mean:.1%} ± {cv_std:.1%}")
    else:
        print("（快速模式沒有準確率數字，想知道請跑完整模式）")
    print("\n接下來可以：")
    print("  python test_model.py        用攝影機測試辨識")
    print("  streamlit run app.py        開啟翻譯網頁")


def main():
    ap = argparse.ArgumentParser(description="一鍵訓練手語辨識模型")
    ap.add_argument("--quick", action="store_true", help="快速模式：跳過交叉驗證")
    ap.add_argument("--yes", "-y", action="store_true", help="不詢問，全部使用預設")
    ap.add_argument("--epochs", type=int, default=60, help="訓練輪數（預設 60）")
    ap.add_argument("--folds", type=int, default=5, help="交叉驗證折數（預設 5）")
    ap.add_argument("--out", default=None,
                    help="模型輸出路徑（預設 data/model.pkl；測試時可指定別的路徑）")
    args = ap.parse_args()

    title("手語辨識模型　一鍵訓練")
    print("這個程式會自動檢查環境、檢查資料，然後訓練模型。")

    title("步驟 A　檢查執行環境")
    check_python()
    check_packages(auto_yes=args.yes)

    title("步驟 B　檢查訓練資料")
    total, n_words = check_data()

    quick = args.quick
    if not quick and not args.yes and IS_TTY:
        est = total * args.folds * args.epochs / 60000
        print(f"\n完整訓練（{args.folds}-fold 交叉驗證 + 最終模型）比較久，"
              f"粗估 {max(est, 1):.0f}～{max(est * 4, 4):.0f} 分鐘（看電腦速度）。")
        if not ask("要跑完整訓練嗎？選 n 則用快速模式（不算準確率，快很多）"):
            quick = True

    if not ask("\n準備好開始訓練了嗎？", auto_yes=args.yes):
        print("已取消。")
        pause_exit(0)

    try:
        run_training(quick, args.epochs, args.folds, args.out)
    except KeyboardInterrupt:
        print("\n\n已中斷訓練（沒有存檔）。")
        pause_exit(1)
    except MemoryError:
        die("記憶體不足。", "關掉其他程式再試，或用 --quick 模式。")
    except Exception as e:
        import traceback
        traceback.print_exc()
        die(f"訓練過程發生錯誤：{type(e).__name__}: {e}",
            "把上面完整錯誤訊息傳給組長。")

    pause_exit(0)


if __name__ == "__main__":
    main()
