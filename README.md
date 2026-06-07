# 台灣手語即時翻譯系統

> 朝陽科技大學 專題作品  
> 結合 MediaPipe 手部關鍵點辨識 + 本地分類模型 + Gemini AI 語意修正

---

## 專題簡介

本系統透過攝影機即時擷取手語動作，利用 MediaPipe 偵測手部關鍵點，搭配本地訓練的機器學習模型進行手語詞彙辨識，最後呼叫 Gemini AI 將辨識出的詞彙序列修飾成自然流暢的中文句子，並透過語音合成朗讀輸出。

### 系統流程

```
攝影機
  ↓
MediaPipe（手部關鍵點擷取）
  ↓
本地分類模型（即時辨識詞彙，無需網路）
  ↓
詞彙緩衝區（累積詞彙）
  ↓ 按下翻譯
Gemini AI（語意修正成自然句子）
  ↓
文字顯示 + 語音朗讀
```

---

## 專案結構

```
sign-language-translator/
├── app.py                  # Streamlit 主程式（入口）
├── collect_data.py         # 錄製手語訓練資料
├── train_model.py          # 訓練本地分類模型
│
├── src/
│   ├── vocab.py            # 詞彙管理（從 JSON 讀取）
│   ├── camera.py           # 攝影機 + MediaPipe 手部偵測
│   ├── gemini_api.py       # Gemini Vision 手語辨識（備用）
│   ├── claude_api.py       # Gemini 語意修正
│   └── tts.py              # Google TTS 語音合成
│
├── data/
│   ├── vocabulary.json     # 詞彙清單（統一管理）
│   ├── keypoints.csv       # 錄製的訓練資料（自動產生）
│   └── model.pkl           # 訓練好的模型（自動產生）
│
├── history/                # 翻譯紀錄（自動產生）
│
├── .env                    # API 金鑰（不可上傳 GitHub）
├── .env.example            # 金鑰範本
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 環境需求

- Python 3.10 以上（建議）
- 有攝影機的電腦
- Gemini API 金鑰（[取得方式](#-api-金鑰取得)）

---

## 安裝步驟

### 1. 安裝套件

```bash
pip install -r requirements.txt
```

### 2. 設定 API 金鑰

```bash
cp .env.example .env
```

用文字編輯器打開 `.env`，填入你的 Gemini API 金鑰：

```
GEMINI_API_KEY=你的金鑰貼這裡
```

### 3. 建立必要資料夾

```bash
mkdir history
```

### 4. 測試 API 連線

```bash
python test_gemini.py
```

出現模型清單表示連線成功。

---

## 使用流程

### Step 1｜錄製手語訓練資料

```bash
python collect_data.py
```

**操作方式：**

| 按鍵 | 動作 |
|------|------|
| 數字 `0`～`9` | 選擇詞彙（對應 vocabulary.json 的順序） |
| `R` | 開始錄製 |
| `S` | 暫停錄製 |
| `Q` | 離開，並顯示各詞彙收集統計 |

> 每個詞彙建議錄製 **100 筆**，光線充足、手放畫面正中央效果最好。

### Step 2｜訓練模型

```bash
python train_model.py
```

訓練完成後會在 `data/` 自動產生 `model.pkl`，並顯示各詞彙的辨識準確率。

### Step 3｜啟動系統

```bash
streamlit run app.py
```

瀏覽器會自動開啟，允許使用攝影機即可開始使用。

**介面操作：**
1. 對著攝影機比手語，系統即時辨識並累積詞彙
2. 辨識到的詞彙顯示在右側「辨識詞彙」區塊
3. 按「翻譯句子」→ Gemini AI 修正成自然中文並播放語音
4. 按「清除重來」→ 重新開始

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

**新增詞彙只需要：**
1. 在 `vocabulary.json` 加入新詞彙
2. 重新執行 `collect_data.py` 錄製新詞彙的資料
3. 重新執行 `train_model.py` 訓練模型

不需要修改任何程式碼。

---

## API 金鑰取得

1. 前往 [Google AI Studio](https://aistudio.google.com/app/apikey)
2. 登入有訂閱 Google AI Pro 的 Google 帳號
3. 點擊「Create API key」
4. 複製金鑰貼入 `.env` 檔案

---

## 常見問題

**攝影機打不開**  
將 `camera.py` 中的 `VideoCapture(0)` 改為 `(1)` 或 `(2)`。

**MediaPipe 安裝失敗（M1/M2 Mac）**  
```bash
pip install mediapipe --no-binary mediapipe
```

**辨識率低**  
- 確認光線充足
- 手部佔畫面比例至少 1/3
- 每個詞彙增加錄製筆數至 200 筆

**Gemini 回傳 429 錯誤**  
API 配額不足，確認 API Key 綁定到付費帳號，或稍等幾分鐘再試。

---

## 📦 套件清單

| 套件 | 用途 |
|------|------|
| streamlit | 網頁介面 |
| mediapipe | 手部關鍵點偵測 |
| google-genai | Gemini API |
| scikit-learn | 本地分類模型 |
| gtts | 語音合成 |
| opencv-python | 影像處理 |
| python-dotenv | 環境變數管理 |

*最後更新：2025*
