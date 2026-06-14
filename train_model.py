import numpy as np
import os, sys, pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.vocab import get_all_labels

# ── 參數 ──────────────────────────────────────
GESTURES  = get_all_labels()
DATA_DIR  = "dynamic_dataset"
SEQ_LEN   = 30
FEAT_DIM  = 126
BATCH     = 32
EPOCHS    = 60
LR        = 1e-3

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ── 讀取資料 ──────────────────────────────────
X, y_raw = [], []
missing  = []

for label in GESTURES:
    folder = Path(DATA_DIR) / label
    files  = list(folder.glob("*.npy")) if folder.exists() else []
    if not files:
        missing.append(label)
        continue
    for f in files:
        seq = np.load(f)                  # (30, 126)
        if seq.shape == (SEQ_LEN, FEAT_DIM):
            X.append(seq)
            y_raw.append(label)

if missing:
    print(f"沒有詞彙：{missing}")

X = np.array(X, dtype=np.float32)        # (N, 30, 126)
print(f"[資料] {len(X)} 筆，{len(set(y_raw))} 個詞彙")

if len(X) == 0:
    print("[錯誤] 沒有資料，請先執行 collect_data.py")
    sys.exit(1)

# 標籤編碼
le      = LabelEncoder()
y       = le.fit_transform(y_raw)        # int array
classes = list(le.classes_)
N_CLASS = len(classes)
print(f"[詞彙] {classes}")

# 切分
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── Dataset ───────────────────────────────────
class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(GestureDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True)
test_loader  = DataLoader(GestureDataset(X_te, y_te), batch_size=BATCH)

# ── LSTM 模型 ─────────────────────────────────
class GestureLSTM(nn.Module):
    def __init__(self, feat_dim, hidden=128, num_layers=2, num_classes=10, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True   # 雙向 LSTM，前後文都參考
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 64),  # 雙向所以 *2
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)       # (batch, seq, hidden*2)
        out = out[:, -1, :]         # 取最後一幀
        out = self.dropout(out)
        return self.fc(out)

model = GestureLSTM(
    feat_dim=FEAT_DIM,
    hidden=128,
    num_layers=2,
    num_classes=N_CLASS
).to(DEVICE)

print(f"\n[模型] 參數量：{sum(p.numel() for p in model.parameters()):,}")

# ── 訓練 ──────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

best_acc  = 0.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    # 訓練
    model.train()
    train_loss, train_correct = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item() * len(xb)
        train_correct += (pred.argmax(1) == yb).sum().item()
    scheduler.step()

    # 驗證
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            val_correct += (model(xb).argmax(1) == yb).sum().item()

    train_acc = train_correct / len(X_tr)
    val_acc   = val_correct   / len(X_te)

    if val_acc > best_acc:
        best_acc   = val_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"loss={train_loss/len(X_tr):.4f}  "
              f"train={train_acc:.1%}  val={val_acc:.1%}  "
              f"best={best_acc:.1%}")

# ── 最終評估 ──────────────────────────────────
model.load_state_dict(best_state)
model.eval()
all_pred, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        all_pred.extend(model(xb).argmax(1).cpu().numpy())
        all_true.extend(yb.numpy())

print("\n[最終結果]")
print(classification_report(all_true, all_pred, target_names=classes))

# ── 儲存 ──────────────────────────────────────
os.makedirs("data", exist_ok=True)
payload = {
    "model_state": best_state,
    "classes":     classes,
    "seq_len":     SEQ_LEN,
    "feat_dim":    FEAT_DIM,
    "hidden":      128,
    "num_layers":  2,
    "best_val_acc": best_acc,
}
with open("data/model.pkl", "wb") as f:
    pickle.dump(payload, f)

print(f"\n[完成] 模型儲存 data/model.pkl  最佳驗證準確率：{best_acc:.1%}")
