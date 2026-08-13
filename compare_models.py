"""A/B 比較：ST-GCN vs LSTM。

在「完全相同的資料、session-aware 切分、增強、正規化」下，
用交叉驗證比較兩個模型的準確率與參數量。

用法：
    python compare_models.py                 # 預設 5-fold、60 epochs
    python compare_models.py --folds 3 --epochs 30
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.trainer import (load_dataset, cross_validate, confusion_pairs,
                         DEVICE, FEAT_DIM, EPOCHS, N_FOLDS)
from src.model import GestureLSTM
from src.model_stgcn import GestureSTGCN

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=EPOCHS)
ap.add_argument("--folds",  type=int, default=N_FOLDS)
ap.add_argument("--min-samples", type=int, default=80,
                help="只比樣本數 >= 此值的詞（過濾尚未收滿的新詞）")
args = ap.parse_args()

X, y, groups, classes = load_dataset(min_samples=args.min_samples)
N_CLASS = len(classes)
print(f"裝置：{DEVICE}　epochs={args.epochs}　folds={args.folds}"
      f"　詞彙數={N_CLASS}（min_samples={args.min_samples}）")

MODELS = {
    "LSTM":   lambda: GestureLSTM(feat_dim=FEAT_DIM, num_classes=N_CLASS).to(DEVICE),
    "ST-GCN": lambda: GestureSTGCN(feat_dim=FEAT_DIM, num_classes=N_CLASS).to(DEVICE),
}

results = {}
for name, factory in MODELS.items():
    n_params = sum(p.numel() for p in factory().parameters())
    print(f"\n========== {name}（參數量 {n_params:,}）==========")
    accs, agg_true, agg_pred = cross_validate(
        factory, X, y, groups, N_CLASS,
        n_folds=args.folds, epochs=args.epochs, log_prefix=f"{name} ")
    print(f"[{name}] {args.folds}-fold：{np.mean(accs):.1%} ± {np.std(accs):.1%}")
    confusion_pairs(agg_true, agg_pred, classes, top=8)
    results[name] = {"accs": accs, "params": n_params}

# ── A/B 總表 ──────────────────────────────────
print("\n================== A/B 比較 ==================")
print(f"{'模型':<8}{'準確率 (mean ± std)':<24}{'參數量':>12}   各折")
for name, r in results.items():
    folds_str = " ".join(f"{a:.0%}" for a in r["accs"])
    print(f"{name:<8}{np.mean(r['accs']):.1%} ± {np.std(r['accs']):.1%}".ljust(32)
          + f"{r['params']:>12,}   [{folds_str}]")

best = max(results, key=lambda n: np.mean(results[n]["accs"]))
diff = (np.mean(results['ST-GCN']['accs']) - np.mean(results['LSTM']['accs'])) * 100
print(f"\n勝出：{best}（ST-GCN − LSTM = {diff:+.1f} 個百分點）")
