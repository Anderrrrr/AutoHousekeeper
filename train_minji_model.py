import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model import LSTMWakeWord
import torch.nn as nn
import torch.optim as optim

# ---------- Dataset ----------
class WakeWordDataset(Dataset):
    def __init__(self, feature_dir):
        self.data = []
        self.labels = []
        self.max_len = 100  # 固定長度，避免 stack error

        for label, folder in enumerate(['background', 'wakeword']):
            path = os.path.join(feature_dir, folder)
            for file in os.listdir(path):
                if file.endswith('.npy'):
                    mfcc = np.load(os.path.join(path, file))
                    self.data.append(mfcc)
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Padding or truncate to fixed length
        if x.shape[0] > self.max_len:
            x = x[:self.max_len, :]
        elif x.shape[0] < self.max_len:
            pad_len = self.max_len - x.shape[0]
            padding = torch.zeros(pad_len, x.shape[1])
            x = torch.cat([x, padding], dim=0)

        return x, y

# ---------- Load Data ----------
dataset = WakeWordDataset("features")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ✅ 檢查資料分布狀況
total_samples = len(dataset)
wakeword_samples = sum(dataset.labels)
background_samples = total_samples - wakeword_samples

print(f"📊 Total samples: {total_samples}")
print(f"🟡 Wakeword samples (Minji): {wakeword_samples}")
print(f"⚫ Background samples: {background_samples}")

# ---------- Model ----------
model = LSTMWakeWord()
criterion = nn.BCELoss()  # 回到最穩定的 BCE loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------- Training ----------
for epoch in range(20):
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        y = y.unsqueeze(1)
        out = model(x)

        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy 計算
        preds = (out > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.size(0)

    accuracy = correct / total
    print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.4f} | Accuracy: {accuracy:.4f}")

# ---------- Save Model ----------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/minji_model.pth")
print("✅ Training complete.")