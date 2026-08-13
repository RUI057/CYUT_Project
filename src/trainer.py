"""共用訓練工具：資料載入、Dataset、交叉驗證、評估。

train_model.py 與 compare_models.py 都用這裡的函式，確保不同模型
在「完全相同的資料、切分、增強、正規化」下比較，A/B 才公平。
模型以 factory（無參數、回傳已在 DEVICE 上的 nn.Module）傳入。
"""
import os
import re
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from src.vocab import get_all_labels
from src.features import (normalize_sequence, augment_sequence, pad_features,
                          FEAT_DIM, HANDS_DIM)

# ── 參數 ──────────────────────────────────────
DATA_DIR     = "dynamic_dataset"
SEQ_LEN      = 30
BATCH        = 32
EPOCHS       = 60
LR           = 1e-3
N_FOLDS      = 5
LABEL_SMOOTH = 0.1

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

_SESSION_RE = re.compile(r"^(\d{8}_\d{6})_")


def load_dataset(min_samples=0):
    """讀取 dynamic_dataset，回傳 (X_raw, y, groups, classes)。

    groups：同詞同次錄製為一組（避免相鄰雙胞胎跨訓練/測試）；舊資料每筆自成一組。
    min_samples：>0 時，樣本數不足的詞會被跳過（用於只比資料齊全的詞）。
    """
    X, y_raw, groups, missing = [], [], [], []
    for label in get_all_labels():
        folder = Path(DATA_DIR) / label
        files  = sorted(folder.glob("*.npy")) if folder.exists() else []
        if not files:
            missing.append(label)
            continue
        for f in files:
            seq = np.load(f)
            # 接受舊 126 維與新 162 維；舊資料臉部補 0 補到 162
            if seq.ndim != 2 or seq.shape[0] != SEQ_LEN or seq.shape[1] not in (HANDS_DIM, FEAT_DIM):
                continue
            X.append(pad_features(seq))
            y_raw.append(label)
            m = _SESSION_RE.match(f.stem)
            groups.append(f"{label}#{m.group(1)}" if m else f"legacy#{f}")

    if missing:
        # 詞彙表有 200 個詞，未收集的通常很多；只印摘要避免洗版
        head = "、".join(missing[:8])
        more = f" ...等 {len(missing)} 個" if len(missing) > 8 else ""
        print(f"[略過] 尚未收集資料的詞：{head}{more}")

    if min_samples > 0:
        cnt  = Counter(y_raw)
        keep = {lab for lab, c in cnt.items() if c >= min_samples}
        dropped = sorted(set(y_raw) - keep, key=lambda l: cnt[l])
        if dropped:
            print(f"[過濾] 樣本數 < {min_samples}，跳過 {len(dropped)} 詞："
                  + "、".join(f"{l}({cnt[l]})" for l in dropped))
        idx    = [i for i, l in enumerate(y_raw) if l in keep]
        X      = [X[i] for i in idx]
        y_raw  = [y_raw[i] for i in idx]
        groups = [groups[i] for i in idx]

    if not X:
        raise SystemExit("[錯誤] 沒有資料，請先執行 collect_data.py")

    X = np.array(X, dtype=np.float32)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = list(le.classes_)
    n_session = sum(1 for g in groups if not g.startswith("legacy#"))
    print(f"[資料] {len(X)} 筆，{len(classes)} 個詞彙"
          f"（含 session 標記 {n_session} 筆 / 舊資料 {len(X)-n_session} 筆）")
    return X, y, np.array(groups), classes


# ── Dataset ───────────────────────────────────
class GestureDataset(Dataset):
    """X 保持 raw；訓練時即時增強，所有資料餵模型前才做尺度正規化。"""
    def __init__(self, X, y, augment=False, seed=0):
        self.X = X
        self.y = y
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        seq = self.X[i]
        if self.augment:
            seq = augment_sequence(seq, self.rng)
        seq = normalize_sequence(seq)
        return (torch.tensor(seq, dtype=torch.float32),
                torch.tensor(self.y[i], dtype=torch.long))


def _class_weights(y_tr, n_class):
    counts = Counter(y_tr.tolist())
    w = torch.tensor([1.0 / counts.get(c, 1) for c in range(n_class)],
                     dtype=torch.float32)
    return (w / w.sum() * n_class).to(DEVICE)


def run_epochs(model, train_loader, val_loader, y_tr, n_class,
               epochs=EPOCHS, log_prefix=""):
    """訓練 epochs 次；有 val 回傳最佳權重，否則回傳最後權重。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(y_tr, n_class),
                                    label_smoothing=LABEL_SMOOTH)
    best_acc, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if val_loader is not None:
            acc = _eval_acc(model, val_loader)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 20 == 0 or epoch == epochs:
                print(f"  {log_prefix}epoch {epoch:3d}/{epochs}  val={acc:.1%}  best={best_acc:.1%}")

    if val_loader is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return best_state, best_acc


def _eval_acc(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total   += len(yb)
    return correct / total


def predict(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
            trues.extend(yb.numpy())
    return preds, trues


def cross_validate(model_factory, X, y, groups, n_class,
                   n_folds=N_FOLDS, epochs=EPOCHS, log_prefix=""):
    """StratifiedGroupKFold 交叉驗證；回傳 (fold_accs, agg_true, agg_pred)。"""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accs, agg_true, agg_pred = [], [], []

    for k, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups), 1):
        tr = DataLoader(GestureDataset(X[tr_idx], y[tr_idx], augment=True, seed=k),
                        batch_size=BATCH, shuffle=True)
        va = DataLoader(GestureDataset(X[va_idx], y[va_idx], augment=False),
                        batch_size=BATCH)
        model = model_factory()
        state, acc = run_epochs(model, tr, va, y[tr_idx], n_class,
                                epochs=epochs, log_prefix=f"{log_prefix}[fold {k}] ")
        model.load_state_dict(state)
        p, t = predict(model, va)
        agg_pred.extend(p); agg_true.extend(t)
        fold_accs.append(acc)
        print(f"{log_prefix}[fold {k}] 最佳驗證準確率：{acc:.1%}")

    return fold_accs, agg_true, agg_pred


def train_full(model_factory, X, y, n_class, epochs=EPOCHS, log_prefix="[final] "):
    """用全部資料訓練最終模型，回傳 state_dict（cpu）。"""
    loader = DataLoader(GestureDataset(X, y, augment=True, seed=0),
                        batch_size=BATCH, shuffle=True)
    model = model_factory()
    state, _ = run_epochs(model, loader, None, y, n_class,
                          epochs=epochs, log_prefix=log_prefix)
    return state


def confusion_pairs(agg_true, agg_pred, classes, top=15):
    """印出最常互相誤判的詞對。"""
    n = len(classes)
    cm = confusion_matrix(agg_true, agg_pred, labels=range(n))
    pairs = [(cm[i, j], classes[i], classes[j])
             for i in range(n) for j in range(n) if i != j and cm[i, j] > 0]
    pairs.sort(reverse=True)
    print("[最常誤判的詞對]  真實 → 被預測成")
    if pairs:
        for cnt, t, p in pairs[:top]:
            print(f"  {t} → {p}：{cnt} 筆")
    else:
        print("  無誤判")


def full_report(agg_true, agg_pred, classes):
    print(classification_report(agg_true, agg_pred, target_names=classes, zero_division=0))
