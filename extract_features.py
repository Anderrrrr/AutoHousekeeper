import os
import librosa
import numpy as np

SAMPLE_RATE = 16000
MAX_LEN = 100  # 對齊模型訓練需求
DATA_DIR = "data"
FEATURE_DIR = "features"
os.makedirs(FEATURE_DIR, exist_ok=True)

for category in ["background", "wakeword"]:
    input_path = os.path.join(DATA_DIR, category)
    output_path = os.path.join(FEATURE_DIR, category)
    os.makedirs(output_path, exist_ok=True)

    files = [f for f in os.listdir(input_path) if f.endswith(".wav")]
    print(f"\n🎧 Extracting features from: {category} ({len(files)} files)")

    for fname in sorted(files):
        try:
            path = os.path.join(input_path, fname)
            audio, _ = librosa.load(path, sr=SAMPLE_RATE)

            # Normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-6)

            # Extract MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13, n_fft=512, hop_length=256)
            mfcc = mfcc.T  # shape: [time, features]

            # Padding or trimming to MAX_LEN
            if mfcc.shape[0] > MAX_LEN:
                mfcc = mfcc[:MAX_LEN, :]
            elif mfcc.shape[0] < MAX_LEN:
                pad_len = MAX_LEN - mfcc.shape[0]
                pad = np.zeros((pad_len, 13))
                mfcc = np.concatenate([mfcc, pad], axis=0)

            outname = os.path.splitext(fname)[0] + ".npy"
            np.save(os.path.join(output_path, outname), mfcc)
            print(f"✅ {fname} → {outname} ({mfcc.shape})")

        except Exception as e:
            print(f"⚠️ Error processing {fname}: {e}")

print("\n✅ Feature extraction complete.")
