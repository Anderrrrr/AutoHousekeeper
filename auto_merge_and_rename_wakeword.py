import os
import shutil
import re

# 資料夾設定
SOURCE_DIR = "data_laptop/wakeword"
TARGET_DIR = "data/wakeword"
os.makedirs(TARGET_DIR, exist_ok=True)

# 找出 target 目前最大編號
def get_max_index(target_dir):
    max_index = 0
    pattern = re.compile(r"wakeword_(\d+)\.wav")
    for fname in os.listdir(target_dir):
        match = pattern.match(fname)
        if match:
            idx = int(match.group(1))
            if idx > max_index:
                max_index = idx
    return max_index

# 合併並重新命名
def merge_files(source_dir, target_dir):
    start_index = get_max_index(target_dir) + 1
    counter = 0

    for fname in os.listdir(source_dir):
        if fname.endswith(".wav"):
            new_name = f"wakeword_{start_index + counter:03d}.wav"
            src_path = os.path.join(source_dir, fname)
            dst_path = os.path.join(target_dir, new_name)
            shutil.copy2(src_path, dst_path)
            print(f"✅ 複製 {fname} → {new_name}")
            counter += 1

    print(f"\n🎉 共合併 {counter} 筆 wakeword 音檔案")

if __name__ == "__main__":
    merge_files(SOURCE_DIR, TARGET_DIR)
