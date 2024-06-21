import torch
from torch import nn


class BasicFullyConnection(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(BasicFullyConnection, self).__init__()
        self.input_size = input_size
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # if x is 3D tensor, catch the last day's feature
        if len(x.shape) == 3:
            x = x[:, -1, :-2]
        x = x.reshape(-1, self.input_size)
        # print(x.shape)
        out = self.fc(x)
        # print(out)
        # exit(1)
        return out

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        return self
