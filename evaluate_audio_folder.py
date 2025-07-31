import os
import librosa
import numpy as np
import torch
from model import LSTMWakeWord

SAMPLE_RATE = 16000
MAX_LEN = 100
AUDIO_FOLDER = "test_audio_samples"  # ⬅️ 你要測試的音檔資料夾
MODEL_PATH = "models/minji_model.pth"

# 載入模型
model = LSTMWakeWord()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# 處理所有音檔
print(f"🔍 Evaluating audio files in: {AUDIO_FOLDER}\n")
files = [f for f in os.listdir(AUDIO_FOLDER) if f.endswith(".wav")]

if not files:
    print("⚠️ 資料夾內沒有任何 .wav 音檔！")
    exit()

for fname in sorted(files):
    path = os.path.join(AUDIO_FOLDER, fname)
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13, n_fft=512, hop_length=256)
    mfcc = mfcc.T  # [time, 13]

    if mfcc.shape[0] > MAX_LEN:
        mfcc = mfcc[:MAX_LEN, :]
    elif mfcc.shape[0] < MAX_LEN:
        pad = np.zeros((MAX_LEN - mfcc.shape[0], 13))
        mfcc = np.concatenate([mfcc, pad], axis=0)

    x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        score = model(x).item()

    print(f"📁 {fname:<30} | Hotword score: {score:.4f}")
