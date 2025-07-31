import sounddevice as sd
import numpy as np
import librosa
import torch
from model import LSTMWakeWord

SAMPLE_RATE = 16000
DURATION = 2  # 錄音長度（秒）
THRESHOLD = 0.5
MAX_LEN = 100

model = LSTMWakeWord()
model.load_state_dict(torch.load("models/minji_model.pth"))
model.eval()

print("🎙️ Recording...")
audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()
audio = audio.flatten()

mean_amp = np.abs(audio).mean()
print(f"📈 Mean amplitude: {mean_amp:.5f}")

if mean_amp < 0.005:
    print("📭 Detected silence, skipping prediction.")
else:
    # Normalize + Amplify
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    audio = audio * 1.5  # Boost gain

    # Optional debug: save audio
    import soundfile as sf
    sf.write("last_record.wav", audio, SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13, n_fft=512, hop_length=256)
    mfcc = mfcc.T

    if mfcc.shape[0] > MAX_LEN:
        mfcc = mfcc[:MAX_LEN, :]
    elif mfcc.shape[0] < MAX_LEN:
        pad = np.zeros((MAX_LEN - mfcc.shape[0], 13))
        mfcc = np.concatenate([mfcc, pad], axis=0)

    x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        score = model(x).item()

    print(f"Hotword probability: {score:.4f}")
    if score > THRESHOLD:
        print("🗣️ Detected MINJI! Responding...")
    else:
        print("🕐 No hotword detected.")
