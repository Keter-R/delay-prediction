import torch
from torch import nn


class LSTM_FEAT(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, num_layers=3, seq_len=1):
        super(LSTM_FEAT, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.seq_len = seq_len
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.7)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * seq_len, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        bath_size = x.size(0)
        h0 = torch.zeros(self.num_layers, bath_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, bath_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(bath_size, -1)
        out = self.fc(out)
        return out
