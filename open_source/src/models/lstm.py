import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        feats = out[:, -1, :]
        return self.head(feats)
