"""訓練 LSTM 手語模型：session-aware 交叉驗證 + 全資料重訓並存檔。

可重用的資料載入 / 訓練 / 評估邏輯集中在 src/trainer.py，
不同模型的比較見 compare_models.py。

用法：
  python train_model.py                 完整訓練（交叉驗證 + 最終模型）
  python train_model.py --quick         跳過交叉驗證，只練出模型
  python train_model.py --epochs 30     調整訓練輪數
"""
import argparse
import os, sys, pickle
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import trainer
from src.trainer import (load_dataset, cross_validate, train_full,
                         confusion_pairs, full_report,
                         DEVICE, EPOCHS, N_FOLDS, SEQ_LEN)
from src.features import FEAT_DIM, MOTION_THRESH
from src.model import GestureLSTM

ap = argparse.ArgumentParser(description="訓練手語辨識模型")
ap.add_argument("--quick", action="store_true", help="跳過交叉驗證（不會有準確率數字）")
ap.add_argument("--epochs", type=int, default=EPOCHS, help=f"訓練輪數（預設 {EPOCHS}）")
ap.add_argument("--folds", type=int, default=N_FOLDS, help=f"交叉驗證折數（預設 {N_FOLDS}）")
ap.add_argument("--min-samples", type=int, default=0, help="樣本數不足的詞直接跳過")
ap.add_argument("--out", default="data/model.pkl", help="模型輸出路徑")
args = ap.parse_args()

CLEAN_TRACKS = trainer.CLEAN_TRACKS
X, y, groups, classes = load_dataset(min_samples=args.min_samples, clean_tracks=CLEAN_TRACKS)
N_CLASS = len(classes)
IN_DIM  = FEAT_DIM

# 每個詞有幾個「錄製組」（同一次錄製為一組，見 trainer.load_dataset）。
# 只有 1 組的詞在 session-aware 交叉驗證時整組被拿去驗證、訓練沒看過，預測必錯：
# 這是評估方式的限制，不代表模型壞掉，補收第二次錄製才能評估。
# 它們不參與準確率、信心門檻校準與誤判詞對的統計，否則必錯的樣本會把信心門檻推到不合理的高。
n_groups  = np.array([len(set(groups[y == c])) for c in range(N_CLASS)])
evaluable = n_groups >= 2
single    = [classes[c] for c in range(N_CLASS) if not evaluable[c]]
factory = lambda: GestureLSTM(feat_dim=IN_DIM, num_classes=N_CLASS).to(DEVICE)

print(f"\n[模型] LSTM  參數量：{sum(p.numel() for p in factory().parameters()):,}"
      f"  輸入 {IN_DIM} 維（雙手）")

# ── 交叉驗證（誠實準確率）──────────────────────
cv_mean = cv_std = 0.0
conf_thr = None
acc_eval = None
report = []     # 交叉驗證結果另存成文字檔：規劃要補收 / 重收哪些詞時要看
if args.quick:
    print("\n[快速模式] 跳過交叉驗證")
else:
    print(f"\n[交叉驗證] {args.folds}-fold（session-aware，舊資料按連號區塊分組）")
    fold_accs, agg_true, agg_pred, agg_proba = cross_validate(
        factory, X, y, groups, N_CLASS, n_folds=args.folds, epochs=args.epochs,
        return_proba=True)
    cv_mean, cv_std = float(np.mean(fold_accs)), float(np.std(fold_accs))
    print(f"\n[交叉驗證結果] {args.folds}-fold 準確率：{cv_mean:.1%} ± {cv_std:.1%}")

    agg_true_a, agg_pred_a = np.asarray(agg_true), np.asarray(agg_pred)
    keep = evaluable[agg_true_a]                     # 只留「可評估的詞」的驗證樣本
    acc_eval = float((agg_pred_a[keep] == agg_true_a[keep]).mean())
    print(f"  排除 {len(single)} 個只有 1 個錄製組、無法評估的詞後（{int(keep.sum())} 筆）：{acc_eval:.1%}")

    print("\n[彙整分類報告]（每筆資料各被驗證一次）")
    report += [f"訓練時間：{datetime.now():%Y-%m-%d %H:%M}",
               f"資料：{len(X)} 筆、{N_CLASS} 個詞；輸入 {IN_DIM} 維（雙手）",
               f"{args.folds}-fold 交叉驗證準確率：{cv_mean:.1%} ± {cv_std:.1%}"
               f"（各折 {' '.join(f'{a:.0%}' for a in fold_accs)}）",
               f"排除 {len(single)} 個只有 1 個錄製組、無法評估的詞後：{acc_eval:.1%}（{int(keep.sum())} 筆）",
               "無法評估的詞（下方 recall=0.00 是評估限制、不代表壞掉；補收第二次錄製才能評估）："
               + ("、".join(single) or "無"), "",
               "[各詞分類報告]（recall 低 = 這個詞常認不出來）",
               full_report(agg_true, agg_pred, classes)]
    report.append("（以下只統計可評估的詞；無法評估的詞預測必錯，列進來會誤導補收方向）")
    report.append(confusion_pairs(agg_true_a[keep], agg_pred_a[keep], classes, top=30))

    # 推理信心門檻：用交叉驗證（模型沒看過的資料）校準，不再寫死
    conf_thr, cover, prec = trainer.calibrate_conf_thresh(agg_proba[keep], agg_true_a[keep])
    line = (f"[信心門檻] 校準為 {conf_thr:.3f}：{cover:.0%} 的預測會出結果，其中 {prec:.0%} 正確"
            f"（目標 {trainer.CONF_TARGET_PRECISION:.0%}）")
    print("\n" + line)
    report += ["", line]

# ── 用全部資料訓練最終模型並儲存 ──────────────
print("\n[最終模型] 以全部資料重新訓練後儲存")
final_state = train_full(factory, X, y, N_CLASS, epochs=args.epochs)

# ── 每個詞的推理動作門檻：靜態手勢（誰、是…）放寬，其餘維持全域門檻 ──
motion_thr = trainer.motion_thresholds(classes)
relaxed = {c: t for c, t in sorted(motion_thr.items(), key=lambda kv: kv[1]) if t < MOTION_THRESH}
print(f"\n[動作門檻] 全域 {MOTION_THRESH}；{len(relaxed)} 個靜態手勢放寬："
      + ("、".join(f"{c}({t:.4f})" for c, t in relaxed.items()) or "無"))

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
payload = {
    "model_state": final_state,
    "classes":     classes,
    "seq_len":     SEQ_LEN,
    "feat_dim":    IN_DIM,
    "clean_tracks": CLEAN_TRACKS,
    "hidden":      128,
    "num_layers":  2,
    "cv_acc_mean": cv_mean,
    "cv_acc_std":  cv_std,
    "cv_acc_evaluable": acc_eval,     # 排除單一錄製組的詞之後的準確率
    "unevaluable_classes": single,
    "motion_thresh": motion_thr,
    "conf_thresh": conf_thr,          # None（--quick 沒有交叉驗證）→ 推理端用預設值
    "label_smooth": trainer.LABEL_SMOOTH,
}
with open(args.out, "wb") as f:
    pickle.dump(payload, f)

if report:
    report_path = os.path.splitext(args.out)[0] + "_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"[報告] 交叉驗證結果與誤判詞對已存到 {report_path}")

print(f"\n[完成] 模型儲存 {args.out}"
      + (f"  交叉驗證準確率：{cv_mean:.1%} ± {cv_std:.1%}" if not args.quick else ""))
