import numpy as np
import os, sys, re, pickle
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.vocab import get_all_labels
from src.features import normalize_sequence, augment_sequence
from src.model import GestureLSTM

# ── 參數 ──────────────────────────────────────
GESTURES   = get_all_labels()
DATA_DIR   = "dynamic_dataset"
SEQ_LEN    = 30
FEAT_DIM   = 126
BATCH      = 32
EPOCHS     = 60
LR         = 1e-3
N_FOLDS    = 5
LABEL_SMOOTH = 0.1

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ── 讀取資料（同時解析 session 分組）──────────────
# 檔名格式：新 = "20260617_143022_5.npy"（含 session）；舊 = "5.npy"（無）
SESSION_RE = re.compile(r"^(\d{8}_\d{6})_")
X, y_raw, groups = [], [], []
missing = []

for label in GESTURES:
    folder = Path(DATA_DIR) / label
    files  = sorted(folder.glob("*.npy")) if folder.exists() else []
    if not files:
        missing.append(label)
        continue
    for f in files:
        seq = np.load(f)                  # (30, 126)
        if seq.shape != (SEQ_LEN, FEAT_DIM):
            continue
        X.append(seq)
        y_raw.append(label)
        m = SESSION_RE.match(f.stem)
        # 同詞同次錄製視為一組（避免相鄰雙胞胎跨訓練/測試）；舊資料每筆自成一組
        groups.append(f"{label}#{m.group(1)}" if m else f"legacy#{f}")

if missing:
    print(f"沒有資料、將跳過的詞彙（{len(missing)}）：{missing}")

X = np.array(X, dtype=np.float32)        # (N, 30, 126)
print(f"[資料] {len(X)} 筆，{len(set(y_raw))} 個詞彙")

if len(X) == 0:
    print("[錯誤] 沒有資料，請先執行 collect_data.py")
    sys.exit(1)

n_session = sum(1 for g in groups if not g.startswith("legacy#"))
print(f"[分組] 含 session 標記 {n_session} 筆 / 舊資料 {len(X)-n_session} 筆"
      f"（舊資料每筆自成一組，誠實切分主要對新收的 session 資料生效）")

# 標籤編碼
le      = LabelEncoder()
y       = le.fit_transform(y_raw)        # int array
groups  = np.array(groups)
classes = list(le.classes_)
N_CLASS = len(classes)


# ── Dataset ───────────────────────────────────
# X 保持 raw（減手腕）；訓練時即時增強，所有資料餵模型前才做尺度正規化
class GestureDataset(Dataset):
    def __init__(self, X, y, augment=False, seed=0):
        self.X = X
        self.y = y
        self.augment = augment
        self.rng = np.random.default_rng(seed)
    def __len__(self):  return len(self.X)
    def __getitem__(self, i):
        seq = self.X[i]
        if self.augment:
            seq = augment_sequence(seq, self.rng)
        seq = normalize_sequence(seq)
        return (torch.tensor(seq, dtype=torch.float32),
                torch.tensor(self.y[i], dtype=torch.long))


# ── LSTM 模型（定義集中於 src/model.py）────────
def make_model():
    return GestureLSTM(feat_dim=FEAT_DIM, hidden=128, num_layers=2,
                       num_classes=N_CLASS).to(DEVICE)


def class_weights(y_tr):
    """樣本少的詞給較高權重，緩解不平衡。"""
    counts = Counter(y_tr.tolist())
    w = torch.tensor([1.0 / counts.get(c, 1) for c in range(N_CLASS)],
                     dtype=torch.float32)
    return (w / w.sum() * N_CLASS).to(DEVICE)


def run_epochs(model, train_loader, val_loader, y_tr, seed=0, log_prefix=""):
    """訓練 EPOCHS 次；有 val 就回傳最佳權重，否則回傳最後權重。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_tr),
                                    label_smoothing=LABEL_SMOOTH)
    best_acc, best_state = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if val_loader is not None:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    correct += (model(xb).argmax(1) == yb).sum().item()
                    total   += len(yb)
            val_acc = correct / total
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 20 == 0 or epoch == EPOCHS:
                print(f"  {log_prefix}epoch {epoch:3d}/{EPOCHS}  val={val_acc:.1%}  best={best_acc:.1%}")

    if val_loader is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return best_state, best_acc


def predict(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
            trues.extend(yb.numpy())
    return preds, trues


# ── 5-fold 交叉驗證（StratifiedGroupKFold：同組不跨 fold）──────
print(f"\n[交叉驗證] {N_FOLDS}-fold（session-aware）")
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_accs = []
agg_true, agg_pred = [], []

for k, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups), 1):
    tr_loader = DataLoader(GestureDataset(X[tr_idx], y[tr_idx], augment=True, seed=k),
                           batch_size=BATCH, shuffle=True)
    va_loader = DataLoader(GestureDataset(X[va_idx], y[va_idx], augment=False),
                           batch_size=BATCH)
    model = make_model()
    state, acc = run_epochs(model, tr_loader, va_loader, y[tr_idx], log_prefix=f"[fold {k}] ")
    model.load_state_dict(state)
    p, t = predict(model, va_loader)
    agg_pred.extend(p); agg_true.extend(t)
    fold_accs.append(acc)
    print(f"[fold {k}] 最佳驗證準確率：{acc:.1%}")

print(f"\n[交叉驗證結果] {N_FOLDS}-fold 準確率："
      f"{np.mean(fold_accs):.1%} ± {np.std(fold_accs):.1%}")

print("\n[彙整分類報告]（每筆資料各被驗證一次）")
print(classification_report(agg_true, agg_pred, target_names=classes, zero_division=0))

# ── 混淆診斷：列出最常互相誤判的詞對 ──────────
cm = confusion_matrix(agg_true, agg_pred, labels=range(N_CLASS))
pairs = [(cm[i, j], classes[i], classes[j])
         for i in range(N_CLASS) for j in range(N_CLASS)
         if i != j and cm[i, j] > 0]
pairs.sort(reverse=True)
print("[最常誤判的詞對]  真實 → 被預測成")
if pairs:
    for cnt, true_w, pred_w in pairs[:15]:
        print(f"  {true_w} → {pred_w}：{cnt} 筆")
else:
    print("  無誤判")

# ── 用全部資料訓練最終模型並儲存 ──────────────
print("\n[最終模型] 以全部資料重新訓練後儲存")
final_model = make_model()
final_state, _ = run_epochs(final_model, DataLoader(
    GestureDataset(X, y, augment=True, seed=0), batch_size=BATCH, shuffle=True),
    val_loader=None, y_tr=y, log_prefix="[final] ")

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
