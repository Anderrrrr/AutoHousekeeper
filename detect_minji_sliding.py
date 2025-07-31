import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)

def main():
    import sounddevice as sd
    import numpy as np
    import librosa
    import torch
    import time
    from collections import deque
    import os
    import sys
    from model import LSTMWakeWord
    from modules.tts_speak import tts_speak
    from modules.center_and_parser import handle_command  # 使用 Google 語音辨識版本

    SAMPLE_RATE = 16000
    SEGMENT_DURATION = 2
    STEP_DURATION = 1
    BUFFER_DURATION = 6
    THRESHOLD = 0.25
    MAX_LEN = 100
    GAIN = 2.0
    COOLDOWN_SECONDS = 3

    model = LSTMWakeWord()
    model.load_state_dict(torch.load("models/minji_model.pth"))
    model.eval()

    print("🔊 開始即時滑動式熱詞偵測（Ctrl+C 停止）")

    audio_buffer = deque(maxlen=int(SAMPLE_RATE * BUFFER_DURATION))
    last_trigger_time = 0

    def audio_callback(indata, frames, time_info, status):
        audio_buffer.extend(indata[:, 0])

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback)
    stream.start()

    try:
        while True:
            if len(audio_buffer) < SAMPLE_RATE * SEGMENT_DURATION:
                time.sleep(0.1)
                print("✅ detect_minji_sliding 還在 loop 中")
                continue

            buffer_array = np.array(audio_buffer)
            buffer_array = buffer_array / (np.max(np.abs(buffer_array)) + 1e-6)
            buffer_array = buffer_array * GAIN

            for start in range(0, SAMPLE_RATE * (BUFFER_DURATION - SEGMENT_DURATION + 1), SAMPLE_RATE * STEP_DURATION):
                segment = buffer_array[start: start + int(SAMPLE_RATE * SEGMENT_DURATION)]
                if np.abs(segment).mean() < 0.005:
                    continue

                mfcc = librosa.feature.mfcc(y=segment, sr=SAMPLE_RATE, n_mfcc=13, n_fft=512, hop_length=256).T
                if mfcc.shape[0] > MAX_LEN:
                    mfcc = mfcc[:MAX_LEN, :]
                elif mfcc.shape[0] < MAX_LEN:
                    pad = np.zeros((MAX_LEN - mfcc.shape[0], 13))
                    mfcc = np.concatenate([mfcc, pad], axis=0)

                x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    score = model(x).item()

                current_time = time.time()
                if score > THRESHOLD and (current_time - last_trigger_time > COOLDOWN_SECONDS):
                    print("🗣️ ✅ Detected MINJI! Responding!")
                    tts_speak("你好，我是 Minji，有什麼我可以幫忙的？")
                    handle_command()
                    last_trigger_time = current_time
                    break

            time.sleep(STEP_DURATION)

    except KeyboardInterrupt:
        stream.stop()
        stream.close()
        print("🛑 偵測中止。")

if __name__ == '__main__':
    main()
