"""共用訓練工具：資料載入、Dataset、交叉驗證、評估。

train_model.py 與 compare_models.py 都用這裡的函式，確保不同模型
在「完全相同的資料、切分、增強、正規化」下比較，A/B 才公平。
模型以 factory（無參數、回傳已在 DEVICE 上的 nn.Module）傳入。
"""
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
from src.features import (augment_sequence, hands_only, to_model_input,
                          clean_hand_tracks, hand_motion, class_motion_thresholds,
                          VALID_DIMS)

# ── 參數 ──────────────────────────────────────
DATA_DIR     = "dynamic_dataset"
SEQ_LEN      = 30
BATCH        = 32
EPOCHS       = 60
LR           = 1e-3
N_FOLDS      = 5
# 標籤平滑關閉：2026-09-28 保留資料實測（90 詞），0.1 與 0 準確率相同（皆 64.9%），
# 但 0.1 會把信心壓扁到分不出對錯（最有把握的前 25% 只有 76% 正確；關閉後 93%），
# 推理端的信心門檻因此失效。信心要能拿來篩選，就不能用標籤平滑。
LABEL_SMOOTH = 0.0

# 推理信心門檻的校準目標：交叉驗證中「信心 ≥ 門檻」的預測要有這麼高的準確率。
# 單一視窗的準確率；推理端還要連續多幀同一個詞才算數，實際更可靠。
CONF_TARGET_PRECISION = 0.85
CONF_FLOOR = 0.3

# 手部軌跡濾波（去幽靈手、補掉幀）：在載入時做，.npy 原始檔不動，
# 調參數不用重錄。推理端依 model.pkl 的 clean_tracks 旗標做同樣處理。
CLEAN_TRACKS = True

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

_SESSION_RE = re.compile(r"^(\d{8}_\d{6})_")

# 舊資料（沒有 session 標記）檔名是連號 0.npy, 1.npy...，當初錄製冷卻只有幾幀，
# 相鄰編號幾乎是同一次動作的複製品。若每筆各自成一組，StratifiedGroupKFold
# 會把雙胞胎拆到訓練/驗證兩邊 → 洩漏、準確率虛高。改成按連號區塊分組。
LEGACY_BLOCK = 10


def load_dataset(min_samples=0, clean_tracks=None):
    """讀取 dynamic_dataset，回傳 (X_raw, y, groups, classes)；X_raw 為 (N, 30, 126) 雙手 raw 特徵。

    groups：同詞同次錄製為一組（避免相鄰雙胞胎跨訓練/測試）；
           舊資料無 session 標記，改按連號區塊（每 LEGACY_BLOCK 筆）分組。
    min_samples：>0 時，樣本數不足的詞會被跳過（用於只比資料齊全的詞）。
    clean_tracks：None 用模組預設 CLEAN_TRACKS；True 會去幽靈手、補掉幀。
    """
    if clean_tracks is None:
        clean_tracks = CLEAN_TRACKS
    X, y_raw, groups, missing = [], [], [], []
    n_tagged = 0
    for label in get_all_labels():
        folder = Path(DATA_DIR) / label
        files  = sorted(folder.glob("*.npy")) if folder.exists() else []
        if not files:
            missing.append(label)
            continue
        for f in files:
            seq = np.load(f)
            # 接受 126 / 162 / 170 維（新舊格式）；舊資料多的臉部、位置欄位丟掉，只留雙手
            if seq.ndim != 2 or seq.shape[0] != SEQ_LEN or seq.shape[1] not in VALID_DIMS:
                continue
            seq = hands_only(seq)
            X.append(clean_hand_tracks(seq) if clean_tracks else seq)
            y_raw.append(label)
            m = _SESSION_RE.match(f.stem)
            if m:
                groups.append(f"{label}#{m.group(1)}")
                n_tagged += 1
            else:
                try:
                    blk = int(f.stem) // LEGACY_BLOCK
                    groups.append(f"{label}#legacy{blk}")
                except ValueError:
                    groups.append(f"legacy#{f}")   # 檔名非連號，保守起見各自成組

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
    print(f"[資料] {len(X)} 筆，{len(classes)} 個詞彙"
          f"（含 session 標記 {n_tagged} 筆 / 舊資料 {len(X)-n_tagged} 筆，"
          f"共 {len(set(groups))} 個切分組）")
    print(f"[手部濾波] {'啟用（去幽靈手、補掉幀）' if clean_tracks else '停用'}")
    return X, y, np.array(groups), classes


def motion_thresholds(classes):
    """每個詞的推理動作門檻（規則見 features.class_motion_thresholds），存進 model.pkl。

    讀原始 .npy、不做手部濾波：推理端是在濾波「前」量動作量，兩邊才對得上。
    """
    motions = {}
    for c in classes:
        ms = []
        for f in (Path(DATA_DIR) / c).glob("*.npy"):
            a = np.load(f)
            if a.ndim == 2 and a.shape[0] == SEQ_LEN and a.shape[1] in VALID_DIMS:
                ms.append(hand_motion(a))
        motions[c] = ms
    return class_motion_thresholds(motions)


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
        seq = to_model_input(seq)
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


def predict(model, loader, return_proba=False):
    model.eval()
    preds, trues, probas = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            pr = torch.softmax(model(xb.to(DEVICE)), dim=1).cpu().numpy()
            preds.extend(pr.argmax(1))
            trues.extend(yb.numpy())
            probas.append(pr)
    if return_proba:
        return preds, trues, np.concatenate(probas)
    return preds, trues


def calibrate_conf_thresh(proba, y_true, target=CONF_TARGET_PRECISION, floor=CONF_FLOOR):
    """用交叉驗證的預測機率，找推理用的信心門檻。

    從最有把握的預測往下累積，取「累積準確率仍 ≥ target」的最寬處，門檻 = 那筆的信心。
    回傳 (門檻, 會出結果的比例, 出結果時的準確率)。達不到 target 時退而求其次，
    取能讓一半預測出結果的門檻，避免門檻高到什麼都認不出來。
    """
    proba, y_true = np.asarray(proba), np.asarray(y_true)
    conf = proba.max(1)
    ok = proba.argmax(1) == y_true
    order = np.argsort(-conf)
    prec = np.cumsum(ok[order]) / np.arange(1, len(ok) + 1)
    min_k = max(1, len(ok) // 20)                  # 至少 5% 的預測，避免被前幾筆的運氣決定
    good = np.flatnonzero(prec[min_k - 1:] >= target)
    k = (good[-1] + min_k) if len(good) else len(ok) // 2
    thr = max(floor, float(conf[order[k - 1]]))
    m = conf >= thr
    return thr, float(m.mean()), float(ok[m].mean()) if m.any() else 0.0


def cross_validate(model_factory, X, y, groups, n_class,
                   n_folds=N_FOLDS, epochs=EPOCHS, log_prefix="",
                   return_proba=False):
    """StratifiedGroupKFold 交叉驗證；回傳 (fold_accs, agg_true, agg_pred)，
    return_proba=True 時多回傳每筆驗證資料的預測機率（給信心門檻校準）。"""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accs, agg_true, agg_pred, agg_proba = [], [], [], []

    for k, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups), 1):
        tr = DataLoader(GestureDataset(X[tr_idx], y[tr_idx], augment=True, seed=k),
                        batch_size=BATCH, shuffle=True)
        va = DataLoader(GestureDataset(X[va_idx], y[va_idx], augment=False),
                        batch_size=BATCH)
        model = model_factory()
        state, acc = run_epochs(model, tr, va, y[tr_idx], n_class,
                                epochs=epochs, log_prefix=f"{log_prefix}[fold {k}] ")
        model.load_state_dict(state)
        p, t, pr = predict(model, va, return_proba=True)
        agg_pred.extend(p); agg_true.extend(t); agg_proba.append(pr)
        fold_accs.append(acc)
        print(f"{log_prefix}[fold {k}] 最佳驗證準確率：{acc:.1%}")

    if return_proba:
        return fold_accs, agg_true, agg_pred, np.concatenate(agg_proba)
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
    """印出最常互相誤判的詞對，並回傳同樣的文字（給訓練報告存檔）。"""
    n = len(classes)
    cm = confusion_matrix(agg_true, agg_pred, labels=range(n))
    pairs = [(cm[i, j], classes[i], classes[j])
             for i in range(n) for j in range(n) if i != j and cm[i, j] > 0]
    pairs.sort(reverse=True)
    lines = ["[最常誤判的詞對]  真實 → 被預測成"]
    lines += [f"  {t} → {p}：{cnt} 筆" for cnt, t, p in pairs[:top]] or ["  無誤判"]
    text = "\n".join(lines)
    print(text)
    return text


def full_report(agg_true, agg_pred, classes):
    """印出各詞的分類報告，並回傳同樣的文字。"""
    text = classification_report(agg_true, agg_pred, target_names=classes, zero_division=0)
    print(text)
    return text
