import sounddevice as sd
import soundfile as sf
import os

SAMPLE_RATE = 16000
DURATION = 2  # 每段錄音秒數
OUTPUT_DIR = "data/background"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🗣 錄製 hard negative 語音樣本，例如：講話但不是 Minji")
print("按 Enter 錄音（每次 2 秒），輸入 q + Enter 結束")

counter = len(os.listdir(OUTPUT_DIR)) + 1

while True:
    cmd = input("👉 按 Enter 錄音（或輸入 q 結束）：")
    if cmd.strip().lower() == "q":
        break

    print("🎙️ 錄音中...請說出一段非 Minji 的語音（例如：蔥抓餅加蛋、我要出門等等）")
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()

    filename = os.path.join(OUTPUT_DIR, f"background_{counter:04d}.wav")
    sf.write(filename, audio, SAMPLE_RATE)
    print(f"💾 儲存：{filename}")
    counter += 1

print("✅ 錄音完成！")
