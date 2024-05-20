import torch
from torch import nn


class BasicFullyConnection(torch.nn.Module):
    def __init__(self, input_size):
        super(BasicFullyConnection, self).__init__()
        self.input_size = input_size
        self.fc = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.reshape(-1, self.input_size)
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
