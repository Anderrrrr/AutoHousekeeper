# AutoHousekeeper｜語音助理原型專案

AutoHousekeeper 是一個結合熱詞偵測（Hotword Detection）、語音識別與語音回應的語音助理原型。此專案以 Python 開發，結合深度學習與聲音處理技術，目標為打造可部署於家庭環境的智慧語音中控系統。

---

## 🚀 專案功能

- 自錄音資料並訓練自定義喚醒詞「Minji」
- 靜態與即時滑動式熱詞辨識
- 基於 MFCC + LSTM 的 PyTorch 模型
- 支援 TTS 回應與語音指令處理模組
- 模組化設計，可擴充控制 Smart Home 元件

---

## 📁 專案結構

\`\`\`
MINJI_HOTWORD_PROJECT/
├── data/                        # 正式訓練用資料
├── data_laptop/                # 原始錄音資料（背景、wakeword）
├── features/                   # 儲存 .npy 格式的 MFCC 特徵
├── models/                     # 訓練完成的模型檔案
├── modules/
│   ├── center_and_parser.py    # 語音指令解析
│   ├── door_checker.py         # 門鎖狀態模擬
│   └── tts_speak.py            # 使用 gTTS 的語音輸出
├── test_audio_samples/         # 測試樣本音檔
├── auto_merge_and_rename.py              # 整理背景音檔案
├── auto_merge_and_rename_wakeword.py     # 整理 wakeword 音檔案
├── check_feature_statistics.py           # 特徵品質檢查
├── detect_back_ground_hard_negative.py   # 錄製 hard negative 資料
├── detect_minji.py                       # 靜態偵測 Minji
├── detect_minji_sliding.py               # 即時滑動式熱詞偵測
├── evaluate_audio_file.py                # 測試單一音檔的預測
├── evaluate_audio_folder.py              # 批次測試資料夾
├── extract_features.py                   # 提取 MFCC 特徵
├── model.py                              # LSTM 模型架構
├── train_minji_model.py                  # 模型訓練流程
└── sliding_last_segment.wav / .mp3       # 測試用音檔
\`\`\`

---

## 🧠 模型架構與訓練

### 🔧 模型：
使用兩層 LSTM 架構搭配全連接輸出層進行二分類（背景音 vs Minji）：

\`\`\`python
class LSTMWakeWord(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=13, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)
\`\`\`

### 🧪 訓練流程：
1. 使用 `.py` 腳本錄製背景音與 wakeword 語音
2. 合併並標準化命名音檔（`auto_merge_and_rename*.py`）
3. 特徵轉換為 MFCC（`extract_features.py`）
4. 執行模型訓練（`train_minji_model.py`）

訓練完成後模型會儲存在 `models/minji_model.pth`

---

## 🗣️ 即時語音助理流程

使用 `detect_minji_sliding.py` 進行即時收音與滑動視窗偵測。

當系統辨識出 "Minji"：

- 使用 `gTTS` 進行語音回應（tts_speak.py）
- 呼叫 `center_and_parser.py` 處理後續語音指令，如：
  - 檢查門鎖狀態（`door_checker.py`）
  - 播放音樂、關閉電視等（可擴充）

---

## 🛠 使用技術

- **程式語言**：Python 3.8+
- **音訊處理**：librosa, sounddevice, soundfile
- **模型訓練**：PyTorch, NumPy
- **TTS**：gTTS（Google Text-to-Speech）
- **自錄資料訓練**：背景音／wakeword 錄製、資料標註、模型訓練完整流程

---

## 📌 注意事項

- 本專案為 Prototype，尚未與實體硬體設備整合
- 不含完整語音 NLP 指令辨識（僅以關鍵字為主）
- 部分模組為簡化或測試版本，實際部署需擴充穩定性

---

## 👨‍💻 作者資訊

鍾牧樺  
國立清華大學｜運動科學 + 資訊工程  
GitHub: [https://github.com/Anderrrrr](https://github.com/Anderrrrr)
