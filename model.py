import torch
import torch.nn as nn

class LSTMWakeWord(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=13, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out) 
