import os
import numpy as np

FEATURE_DIR = "features"

for category in ["background", "wakeword"]:
    path = os.path.join(FEATURE_DIR, category)
    files = [f for f in os.listdir(path) if f.endswith(".npy")]
    print(f"\n📂 {category.upper()} - {len(files)} samples")

    if not files:
        print("⚠️ No features found.")
        continue

    all_shapes = []
    all_means = []
    all_max = []
    all_min = []

    for f in files:
        data = np.load(os.path.join(path, f))
        all_shapes.append(data.shape)
        all_means.append(np.mean(data))
        all_max.append(np.max(data))
        all_min.append(np.min(data))

    print(f"🔸 Shape - min: {min(all_shapes)}, max: {max(all_shapes)}")
    print(f"🔸 Mean  - avg: {np.mean(all_means):.5f}, min: {np.min(all_means):.5f}, max: {np.max(all_means):.5f}")
    print(f"🔸 Max   - avg: {np.mean(all_max):.5f}, max: {np.max(all_max):.5f}")
    print(f"🔸 Min   - avg: {np.mean(all_min):.5f}, min: {np.min(all_min):.5f}")
