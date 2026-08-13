# 台灣手語即時辨識系統

手語詞彙識別與自然語言轉換系統，使用深度學習進行即時的手語動作識別。

## 概述

本項目結合 MediaPipe 手部關鍵點檢測、PyTorch LSTM 時間序列模型及 Gemini API，建構一套從手語捕捉、特徵提取、詞彙識別到自然語言轉換的完整流程。系統支持本地訓練與推理，同時提供文字轉語音功能。

## 核心功能

- 攝影機實時手語捕捉與特徵提取
- 基於 LSTM 的時間序列詞彙識別
- Gemini API 語義修正與自然句子生成
- Google TTS 文字語音合成
- 訓練資料的自動收集與管理

## 項目結構

```
CYUT_MDFK/
├── app.py                    網頁即時翻譯
├── collect_data.py           收集訓練資料
├── train_model.py            LSTM 模型訓練
├── test_model.py             實時識別測試
├── hand_landmarker.task      MediaPipe 模型文件
├── requirements.txt          Python 依賴
├── data/
│   ├── vocabulary.json       詞彙列表
│   ├── model.pkl             訓練後的模型
│   └── 手語演示網址.txt
├── dynamic_dataset/          訓練數據目錄（按詞彙分類）
│   ├── 工作/
│   ├── 你好/
│   ├── 謝謝/
│   └── ... (25 個詞彙)
├── history/                  識別紀錄
│   └── records.json
└── src/
    ├── camera.py             攝影機推理與穩定確認 (SignRecognizer)
    ├── model.py              LSTM 模型定義（train/test/camera 共用）
    ├── features.py           特徵抽取、尺度正規化與資料增強
    ├── vocab.py              詞彙管理
    ├── gemini_api.py         Gemini API 接口
    ├── tts.py                文字轉語音
    └── font_utils.py         文字繪製工具
```

## 工作流程

### 數據收集

```bash
python collect_data.py
```

選定詞彙後，手部穩定出現即自動連續錄製每筆 30 幀序列。按鍵控制：
- N / M: 下一個 / 上一個詞彙
- R: 重置目前詞彙的所有資料
- D: 刪除上一筆
- S: 暫停 / 繼續偵測（切換詞彙時會自動暫停，避免誤觸）
- Q: 離開

### 模型訓練

```bash
python train_model.py
```

使用 dynamic_dataset 中的訓練數據訓練 LSTM 模型。訓練參數：
- 序列長度: 30 幀
- 特徵維度: 126 (雙手 × 21 關鍵點 × 3 軸 xyz)
- 隱藏層: 128 單位，2 層雙向
- 批次: 32
- 訓練周期: 60
- 學習率: 1e-3

訓練時自動套用尺度正規化、資料增強（時間伸縮 / 旋轉 / 抖動）與類別權重（緩解樣本不平衡）。
訓練結束會印出 classification report 與「最常誤判的詞對」，並生成 data/model.pkl。

### 實時測試

```bash
python test_model.py
```

啟動攝影機進行即時識別（OpenCV 視窗）。

### 網頁即時翻譯

```bash
streamlit run app.py
```

1. 對著鏡頭比手語，系統即時辨識並累積詞彙
2. 手離開畫面（停頓）→ 自動將詞彙序列送 Gemini 轉成自然中文句子
3. gTTS 朗讀句子，並保存到翻譯紀錄
4. 也可手動按「翻譯 / 朗讀 / 清除」，左側側邊欄可調信心門檻與確認幀數

## 環境要求

- Python 3.10+
- 攝影機設備
- Gemini API 密鑰（網頁語義翻譯需要；純辨識測試 test_model.py 可不用）
- GPU 支持加速（CUDA 或 Apple Silicon MPS），可選

## 安裝與配置

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 配置環境變量

創建 .env 文件：

```
GEMINI_API_KEY=your_api_key_here
```

### 3. 準備目錄

```bash
mkdir -p history
```

## 技術細節

### 特徵提取

使用 MediaPipe 檢測雙手各 21 個關鍵點的 (x, y, z) 座標，相對手腕做平移正規化（左手 63 維 + 右手 63 維 = 126 維）。餵入模型前再對每隻手做尺度正規化（除以手部大小），消除離鏡頭遠近造成的誤判。特徵抽取與正規化集中於 `src/features.py`，收集 / 訓練 / 推理三端共用以確保一致。

### 模型架構

雙向 LSTM 用于捕捉時間序列中的手語動作特征：
- 輸入: (batch_size, 30, 126)
- 雙向 LSTM: 128 隱藏單位，2 層（輸出 256 維 / 幀）
- 特徵融合: 時間維度平均池化 + 最大池化（256 × 2 = 512 維）
- 全連接層: 512 -> 64 (ReLU) -> num_classes

### 詞彙管理

vocabulary.json 定義所有可識別的詞彙，修改后需重新收集數據並訓練模型。

## 詞彙列表

當前支持 25 個詞彙，存儲在 dynamic_dataset 目錄中：

工作、不好、不要、他、去、再見、吃飯、名字、好、你、你好、我、沒關係、來、看醫生、要、喜歡、喝水、睡覺、對不起、請、學校、錢、幫忙、謝謝

## 依賴包

| 包名 | 功能 |
|------|------|
| torch | 深度學習框架 |
| mediapipe | 手部關鍵點檢測 |
| matplotlib | mediapipe 繪圖相依（必裝） |
| opencv-python | 圖像處理與攝影機控制 |
| google-genai | Gemini API 調用 |
| gtts | 文字語音合成 |
| numpy | 數值計算 |
| scikit-learn | 數據預處理 |
| pandas | 數據處理 |
| pillow | 圖像操作 |
| python-dotenv | 環境變量管理 |
| streamlit | 網頁即時翻譯介面 |

## 使用建議

- 每個詞彙至少收集 100 筆數據
- 確保光線充足
- 手部應佔屏幕至少 1/3
- 在多種環境下測試以提高泛化性能

## 性能參考

基于 25 詞彙、每詞 100-130 筆訓練數據：
- 識別準確率: 85-90%
- 推理延遲: 50-100ms (GPU) / 200-300ms (CPU)
- 內存占用: 500MB - 2GB

## 更新紀錄

2026.6.14 - 引入深度學習，採用 PyTorch + Bi-LSTM 
2026.6.15 - 尺度正規化 + 資料增強 + 類別權重 + mean/max 池化，降低手勢誤判

## 未來計劃

- 擴展詞彙數量
- 本地化語義修正模型，減少 API 調用
- 全身姿勢識別支持
- 離線模式支持