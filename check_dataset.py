"""資料品質健檢：掃描 dynamic_dataset 找出可疑樣本。

檢查項目：
1. shape 錯誤     — 不是 (30, 126)，訓練時會被靜默跳過
2. 空樣本         — 過多幀完全沒偵測到手（錄到空景）
3. 靜止樣本       — 動作幅度 < MOTION_THRESH，即時推理根本不會觸發預測
4. 掉手樣本       — 該詞多數樣本是雙手，此樣本卻大半時間只有單手
5. 離群樣本       — 正規化後的平均姿態離同詞中心 > mean+3σ

只列清單不刪除；確認後可用印出的指令刪。
用法：python check_dataset.py [--data-dir dynamic_dataset]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.features import normalize_sequence, pad_features, HAND_DIM, HANDS_DIM, FEAT_DIM

SEQ_LEN = 30
MOTION_THRESH   = 0.003   # 與 SignRecognizer 一致（小幅度手勢如「謝謝」中位僅 ~0.004）
EMPTY_FRAME_PCT = 0.30    # 超過 30% 幀沒手 → 空樣本
TWO_HAND_CLASS  = 0.70    # 全詞 both-hands 幀比 > 70% → 視為雙手詞
DROPOUT_SAMPLE  = 0.30    # 雙手詞中樣本 both-hands 幀比 < 30% → 掉手
OUTLIER_SIGMA   = 3.0

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="dynamic_dataset")
args = ap.parse_args()

root = Path(args.data_dir)
labels = sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))

issues = {"shape": [], "empty": [], "static": [], "dropout": [], "outlier": []}
per_word = {}

for label in labels:
    files = sorted((root / label).glob("*.npy"))
    if not files:
        continue

    feats, metas = [], []
    both_fracs = []
    for f in files:
        seq = np.load(f)
        # 接受舊 126 維與新 162 維；其餘視為壞檔
        if seq.ndim != 2 or seq.shape[0] != SEQ_LEN or seq.shape[1] not in (HANDS_DIM, FEAT_DIM):
            issues["shape"].append((label, f.name, f"shape={seq.shape}"))
            continue
        seq = pad_features(seq)

        L = np.abs(seq[:, :HAND_DIM]).sum(1) > 1e-6           # 每幀左手存在
        R = np.abs(seq[:, HAND_DIM:HANDS_DIM]).sum(1) > 1e-6  # 右手（不含臉部）
        any_hand  = (L | R).mean()
        both_hand = (L & R).mean()
        motion    = float(np.std(seq[:, :HANDS_DIM], axis=0).mean())  # 只算手部動作
        has_face  = bool((np.abs(seq[:, HANDS_DIM:]).sum(1) > 1e-6).any())

        metas.append((f, any_hand, both_hand, motion, has_face))
        both_fracs.append(both_hand)
        feats.append(normalize_sequence(seq).mean(0))     # 平均姿態 (126,)

    if not metas:
        continue

    is_two_hand = np.mean(both_fracs) > TWO_HAND_CLASS
    flag_e = flag_s = flag_d = 0

    for f, any_hand, both_hand, motion, _hf in metas:
        if (1 - any_hand) > EMPTY_FRAME_PCT:
            issues["empty"].append((label, f.name, f"{(1-any_hand):.0%} 幀沒手"))
            flag_e += 1
        if motion < MOTION_THRESH:
            issues["static"].append((label, f.name, f"motion={motion:.4f}"))
            flag_s += 1
        if is_two_hand and both_hand < DROPOUT_SAMPLE:
            issues["dropout"].append((label, f.name, f"雙手幀僅 {both_hand:.0%}"))
            flag_d += 1

    # 離群：到同詞中心的距離 > mean + 3σ
    F = np.stack(feats)
    centroid = F.mean(0)
    dists = np.linalg.norm(F - centroid, axis=1)
    thr = dists.mean() + OUTLIER_SIGMA * dists.std()
    flag_o = 0
    for (f, *_), d in zip(metas, dists):
        if d > thr:
            issues["outlier"].append((label, f.name, f"dist={d:.2f} (thr={thr:.2f})"))
            flag_o += 1

    n_face = sum(1 for m in metas if m[4])
    per_word[label] = dict(n=len(metas), two_hand=is_two_hand, face=n_face,
                           empty=flag_e, static=flag_s, dropout=flag_d, outlier=flag_o)

# ── 報告 ──────────────────────────────────────
print(f"{'詞彙':<8}{'筆數':>5}{'含臉':>6}{'雙手':>5}{'空':>4}{'靜止':>5}{'掉手':>5}{'離群':>5}")
print("-" * 47)
total_bad = 0
for label, s in per_word.items():
    bad = s["empty"] + s["static"] + s["dropout"] + s["outlier"]
    total_bad += bad
    mark = " ←" if bad else ""
    print(f"{label:<8}{s['n']:>5}{s['face']:>6}{'是' if s['two_hand'] else '　':>5}"
          f"{s['empty']:>4}{s['static']:>5}{s['dropout']:>5}{s['outlier']:>5}{mark}")

names = {"shape": "shape 錯誤", "empty": "空樣本", "static": "靜止樣本",
         "dropout": "掉手樣本", "outlier": "離群樣本"}
print()
for key, items in issues.items():
    if not items:
        continue
    print(f"[{names[key]}] {len(items)} 筆")
    for label, fname, detail in items[:20]:
        print(f"  {label}/{fname}  {detail}")
    if len(items) > 20:
        print(f"  ...（其餘 {len(items)-20} 筆）")
    print()

# 臉部覆蓋摘要
tot   = sum(s["n"] for s in per_word.values())
tface = sum(s["face"] for s in per_word.values())
w_none = [l for l, s in per_word.items() if s["face"] == 0]
w_some = [l for l, s in per_word.items() if 0 < s["face"] < s["n"]]
print(f"[臉部覆蓋] {tface}/{tot} 筆含臉部資料（{tface/tot:.0%}）")
if w_none:
    print(f"  完全沒有含臉樣本的詞（{len(w_none)}）：" + "、".join(w_none[:12])
          + (" ..." if len(w_none) > 12 else ""))
    print("  ⚠️ 這些詞若不補收含臉樣本，模型可能靠『有沒有臉』作弊，訓練分數會虛高。")
if w_some:
    print(f"  部分含臉的詞（{len(w_some)}）：" + "、".join(w_some[:12]))
print()

n_issues = sum(len(v) for v in issues.values())
print(f"總計：{n_issues} 筆可疑（部分樣本可能同時中多項）")
if n_issues:
    out = Path("data/quality_report.txt")
    with out.open("w", encoding="utf-8") as fh:
        for key, items in issues.items():
            for label, fname, detail in items:
                fh.write(f"{key}\t{label}/{fname}\t{detail}\n")
    print(f"完整清單已寫入 {out}（確認後可依清單手動刪除，再重新訓練）")
