import numpy as np
import os
import sys
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.vocab import get_all_labels

GESTURES  = get_all_labels()
DATA_DIR  = "dynamic_dataset"
SEQ_LEN   = 30   # 每筆幾幀
FEAT_DIM  = 126  # 每幀特徵維度（雙手 63×2）

print(f"[詞彙] 共 {len(GESTURES)} 個：{GESTURES}\n")

X, y = [], []
missing = []

for label in GESTURES:
    folder = os.path.join(DATA_DIR, label)
    files  = [f for f in os.listdir(folder) if f.endswith(".npy")] if os.path.exists(folder) else []

    if not files:
        missing.append(label)
        continue

    for fname in files:
        seq = np.load(os.path.join(folder, fname))  # shape: (30, 126)
        # 展平成一維向量 30×126 = 3780 維，給 Random Forest 用
        X.append(seq.flatten())
        y.append(label)

if missing:
    print(f"[跳過] 以下詞彙尚無資料：{missing}\n")

X = np.array(X)
y = np.array(y)
print(f"[資料] 共 {len(X)} 筆，{len(set(y))} 個詞彙\n")

if len(X) == 0:
    print("[錯誤] 沒有任何訓練資料，請先執行 collect_data.py")
    sys.exit(1)

# 訓練
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("[訓練] Random Forest 訓練中...")
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 評估
print("\n[結果] 測試集準確率：")
print(classification_report(y_test, model.predict(X_test)))

# 儲存
payload = {
    "model":    model,
    "labels":   list(set(y)),
    "seq_len":  SEQ_LEN,
    "feat_dim": FEAT_DIM
}
os.makedirs("data", exist_ok=True)
with open("data/model.pkl", "wb") as f:
    pickle.dump(payload, f)

print("[完成] 模型已儲存到 data/model.pkl")
