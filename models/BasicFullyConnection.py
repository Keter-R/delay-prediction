import torch
from torch import nn


class BasicFullyConnection(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(BasicFullyConnection, self).__init__()
        self.input_size = input_size
        print('BFC liner input size:', input_size)
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
        # print(x.shape)
        # print(self.input_size)
        # exit(13)
        # x = x.reshape(-1, self.input_size)
        # print(x.shape)
        out = self.fc(x)
        # print(out)
        # exit(1)
        return out

    # def weight_init(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight)
    #             nn.init.constant_(m.bias, 0)
    #     return self


class FCForGraph(nn.Module):
    def __init__(self, feature_num, hidden_size=128):
        super(FCForGraph, self).__init__()
        self.feature_num = feature_num
        print('FCFG liner input size:', feature_num - 1)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(feature_num - 1),
            nn.Linear(feature_num - 1, hidden_size),  # exclude station id
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = x[:, -self.feature_num:-1]
        x = self.fc(x)
        return x
