"""訓練 LSTM 手語模型：session-aware 交叉驗證 + 全資料重訓並存檔。

可重用的資料載入 / 訓練 / 評估邏輯集中在 src/trainer.py，
不同模型的比較見 compare_models.py。
"""
import os, sys, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.trainer import (load_dataset, cross_validate, train_full,
                         confusion_pairs, full_report,
                         DEVICE, EPOCHS, N_FOLDS, SEQ_LEN, FEAT_DIM)
from src.model import GestureLSTM

X, y, groups, classes = load_dataset()
N_CLASS = len(classes)
factory = lambda: GestureLSTM(feat_dim=FEAT_DIM, num_classes=N_CLASS).to(DEVICE)

print(f"\n[模型] LSTM  參數量：{sum(p.numel() for p in factory().parameters()):,}")

# ── 交叉驗證（誠實準確率）──────────────────────
print(f"\n[交叉驗證] {N_FOLDS}-fold（session-aware）")
fold_accs, agg_true, agg_pred = cross_validate(factory, X, y, groups, N_CLASS)
print(f"\n[交叉驗證結果] {N_FOLDS}-fold 準確率："
      f"{np.mean(fold_accs):.1%} ± {np.std(fold_accs):.1%}")

print("\n[彙整分類報告]（每筆資料各被驗證一次）")
full_report(agg_true, agg_pred, classes)
confusion_pairs(agg_true, agg_pred, classes)

# ── 用全部資料訓練最終模型並儲存 ──────────────
print("\n[最終模型] 以全部資料重新訓練後儲存")
final_state = train_full(factory, X, y, N_CLASS)

os.makedirs("data", exist_ok=True)
payload = {
    "model_state": final_state,
    "classes":     classes,
    "seq_len":     SEQ_LEN,
    "feat_dim":    FEAT_DIM,
    "hidden":      128,
    "num_layers":  2,
    "cv_acc_mean": float(np.mean(fold_accs)),
    "cv_acc_std":  float(np.std(fold_accs)),
}
with open("data/model.pkl", "wb") as f:
    pickle.dump(payload, f)

print(f"\n[完成] 模型儲存 data/model.pkl"
      f"  交叉驗證準確率：{np.mean(fold_accs):.1%} ± {np.std(fold_accs):.1%}")
