"""手語辨識模型定義
雙向 LSTM + 時間維度 mean/max 池化。
"""
import torch
import torch.nn as nn


class GestureLSTM(nn.Module):
    def __init__(self, feat_dim, hidden=128, num_layers=2, num_classes=10, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers, batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 4, 64),  # 雙向(*2) × (mean+max 池化)(*2)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        feat = torch.cat([out.mean(1), out.max(1).values], dim=1)
        return self.fc(self.dropout(feat))
