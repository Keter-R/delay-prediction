import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_len=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.7)
        # 2-type classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * seq_len, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(x.size(0), -1)
        out = self.classifier(out)
        out = out.squeeze(1)
        return out
