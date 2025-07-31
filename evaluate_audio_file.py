import librosa
import numpy as np
import torch
from model import LSTMWakeWord

SAMPLE_RATE = 16000
MAX_LEN = 100
AUDIO_PATH = "sliding_last_segment.wav"  # <- 你要分析的檔案

model = LSTMWakeWord()
model.load_state_dict(torch.load("models/minji_model.pth"))
model.eval()

# 載入音檔
audio, _ = librosa.load(AUDIO_PATH, sr=SAMPLE_RATE)
audio = audio / (np.max(np.abs(audio)) + 1e-6)

# MFCC 特徵
mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13, n_fft=512, hop_length=256).T
if mfcc.shape[0] > MAX_LEN:
    mfcc = mfcc[:MAX_LEN, :]
elif mfcc.shape[0] < MAX_LEN:
    pad = np.zeros((MAX_LEN - mfcc.shape[0], 13))
    mfcc = np.concatenate([mfcc, pad], axis=0)

# 預測分數
x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    score = model(x).item()

print(f"🎧 Hotword score of {AUDIO_PATH}: {score:.4f}")
