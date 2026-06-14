# 台灣手語即時翻譯系統

> 朝陽科技大學 專題作品
> 結合 MediaPipe 手部關鍵點擷取 + PyTorch LSTM 時間序列模型 + Gemini AI 語意修正

---

## 專題簡介

本系統透過攝影機擷取手語動作，使用 MediaPipe 偵測手部關鍵點，搭配本地訓練的 LSTM 時間序列模型辨識詞彙，並透過 Gemini AI 將詞彙序列修飾成自然中文句子，最後使用 Google TTS 進行語音朗讀。

### 系統流程

```
攝影機
  ↓
MediaPipe（手部關鍵點擷取）
  ↓
特徵序列（126維度 x 30幀）
  ↓
PyTorch LSTM（即時辨識詞彙）
  ↓
詞彙緩衝區（累積20幀確認）
  ↓
Gemini AI（句子修飾 + 語意理解）
  ↓
Google TTS（文字轉語音）
  ↓
螢幕顯示 + 語音朗讀
```

---

## 專案結構

```
CYUT_MDFK/
├── collect_data.py         # 錄製手語訓練資料
├── train_model.py          # 訓練 PyTorch LSTM 模型
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
    ├── camera.py           # 本地辨識與手部特徵萃取
    ├── gemini_api.py       # Gemini API 語意修正
    ├── tts.py              # 語音合成模組
    ├── vocab.py            # 詞彙管理工具
    └── font_utils.py       # 文字繪製工具
```

---

## 環境需求

- Python 3.10 以上（建議）
- 有攝影機的電腦
- 支援 GPU 加速更佳（支援 CUDA 和 Apple Silicon MPS）
- Google Gemini API 金鑰（如果要使用語意修正功能）

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

## 技術特點

- **即時處理**：30幀序列 (約1秒) 進行一次推理
- **自動確認**：連續20幀正確判斷才確認詞彙
- **GPU 加速**：支援 CUDA、Apple Silicon (MPS)、CPU 自動選擇
- **特徵工程**：126維度手部關鍵點特徵 (21個關鍵點 × 6 = 126)
- **序列模型**：雙向 LSTM (128 hidden units, 2 layers)

### 模型架構

```
輸入層 (30, 126)
  ↓
Bidirectional LSTM
  - Hidden: 128 units
  - Layers: 2
  - Dropout: 0.3
  ↓
Dropout (0.3)
  ↓
全連接層
  - FC1: 128 → 64 (ReLU)
  - Dropout: 0.3
  - FC2: 64 → num_classes
  ↓
輸出層 (softmax probability)
```

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

**辨識率低**
- 確認光線充足
- 手部佔畫面比例至少 1/3
- 每個詞彙增加錄製筆數

---

## 套件清單

| 套件 | 用途 |
|------|------|
| torch | 深度學習框架 (LSTM 時間序列模型) |
| mediapipe | 手部關鍵點偵測 |
| opencv-python | 影像處理與攝影機控制 |
| google-genai | Gemini API 語意修正 |
| gtts | 文字轉語音 |
| streamlit | 網頁應用框架 (可選) |
| python-dotenv | 環境變數管理 |
| pillow | 圖像處理 |
| numpy | 數值計算 |
| scikit-learn | 資料預處理 |
| pandas | 資料處理 |

---

## 效能指標 (參考)

基於訓練資料集 (25 個詞彙，每個 100-130 筆)：

- **辨識準確率**：~85-90% (依詞彙清晰度)
- **推理延遲**：50-100 ms (GPU) / 200-300 ms (CPU)
- **記憶體佔用**：500 MB - 2 GB (依設備)

> 實際效能取決於：手部清晰度、光線條件、訓練資料品質

---

## 更新

```bash
    2026.6.14
      修改成動態辨識 導入深度學習 
      PyTorch + Bi-LSTM
```
---

## 未來改進計劃

- [ ] 使用爬蟲技術整合台灣手語資料庫進行訓練
- [ ] 加入更多詞彙 (目前 25 個)
- [ ] 語意修飾模型本地化（減少 API 調用）
- [ ] 詞彙斷句與自動觸發機制
- [ ] 整合 MediaPipe Holistic（全身姿勢識別）
- [ ] 使用 Docker 容器化部署
- [ ] 網頁應用界面 (基於 Streamlit)
- [ ] 離線模式支援
- [ ] 多手語言支援