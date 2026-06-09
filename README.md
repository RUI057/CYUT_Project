# 台灣手語即時翻譯系統

> 朝陽科技大學 專題作品
> 結合 MediaPipe 手部關鍵點擷取 + 本地分類模型 + Gemini AI 語意修正

---

## 專題簡介

本系統透過攝影機擷取手語動作，使用 MediaPipe 偵測手部關鍵點，搭配本地訓練模型辨識詞彙，並透過 Gemini AI 將詞彙序列修飾成自然中文句子。

### 系統流程

```
攝影機
  ↓
MediaPipe（手部關鍵點擷取）
  ↓
本地分類模型（即時辨識詞彙）
  ↓
詞彙緩衝區（累積詞彙）
  ↓
Gemini AI（句子修飾）
  ↓
文字顯示 + 語音朗讀
```

---

## 專案結構

```
CYUT_MDFK/
├── collect_data.py         # 錄製手語訓練資料
├── train_model.py          # 訓練本地分類模型
├── test_model.py           # 即時辨識測試
├── README.md
├── requirements.txt
├── data/
│   ├── vocabulary.json     # 詞彙清單
│   ├── keypoints.csv       # 訓練資料紀錄檔
│   └── model.pkl           # 訓練好的模型
├── dynamic_dataset/        # 手語訓練資料資料夾
├── history/                # 翻譯紀錄
└── src/
    ├── camera.py          # 本地辨識與手部特徵萃取
    ├── gemini_api.py      # Gemini API 語意修正
    ├── tts.py             # 語音合成模組
    ├── vocab.py           # 詞彙管理工具
    └── font_utils.py      # 文字繪製工具
```

---

## 環境需求

- Python 3.10 以上（建議）
- 有攝影機的電腦
- Gemini API 金鑰（如果要使用 `src/gemini_api.py`）

---

## 安裝步驟

### 1. 安裝套件

```bash
pip install -r requirements.txt
```

### 2. 設定環境變數

建立 `.env` 檔案，並加入：

```bash
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
```

### 3. 建立必要資料夾

```bash
mkdir history
```

---

## 使用流程

### Step 1｜錄製手語訓練資料

```bash
python collect_data.py
```

**操作方式：**

| 按鍵 | 動作 |
|------|------|
| 數字 `0`～`9` | 選擇詞彙（對應 `data/vocabulary.json` 的順序） |
| `R` | 開始錄製 |
| `S` | 暫停錄製 |
| `D` | 刪除上一筆訓練資料｜
| `Q` | 離開 |

> 建議每個詞彙錄製 100 筆以上，光線充足且手放畫面中央效果較好。

### Step 2｜訓練模型

```bash
python train_model.py
```

訓練完成後會在 `data/model.pkl` 產生模型檔案。

### Step 3｜測試模型

```bash
python test_model.py
```

系統會啟動攝影機，並即時顯示辨識結果。

---

## 詞彙管理

所有詞彙統一在 `data/vocabulary.json` 管理：

```json
{
  "groups": [
    {
      "id": 1,
      "name": "基本問候",
      "priority": "高",
      "labels": ["你好", "再見", "謝謝", "對不起", "沒關係", "請"]
    }
  ]
}
```

新增詞彙流程：

1. 編輯 `data/vocabulary.json`
2. 重新執行 `python collect_data.py` 錄製新詞彙資料
3. 重新執行 `python train_model.py` 訓練模型

---

## API 金鑰取得

1. 前往 [Google AI Studio](https://aistudio.google.com/app/apikey)
2. 登入支援的 Google 帳號
3. 建立 API Key
4. 將金鑰貼入 `.env`

---

## 常見問題

**攝影機打不開**
將 `src/camera.py` 中的 `cv2.VideoCapture(0)` 改為 `(1)` 或 `(2)`。

**MediaPipe 安裝失敗（M1/M2 Mac）**

```bash
pip install mediapipe --no-binary mediapipe
```

**辨識率低**
- 確認光線充足
- 手部佔畫面比例至少 1/3
- 每個詞彙增加錄製筆數

---

## 套件清單

| 套件 | 用途 |
|------|------|
| scikit-learn | 本地分類模型 |
| gtts | 語音合成 |
| opencv-python | 影像處理 |
| python-dotenv | 環境變數管理 |
