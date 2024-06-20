import torch
from torch import nn
import models.BasicFullyConnection as BasicFullyConnection
import models.LSTM_FEAT as LSTM_FEAT


class FCLstm(nn.Module):
    def __init__(self, feature_num, seq_len, seq_feature_num=128, hidden_size=128, num_layers=3):
        super(FCLstm, self).__init__()
        self.feature_num = feature_num
        self.seq_len = seq_len
        self.seq_feature_num = seq_feature_num
        self.lstm = LSTM_FEAT.LSTM_FEAT(feature_num, seq_feature_num, seq_len=seq_len - 1)
        self.fc = nn.Sequential(
            nn.Linear(feature_num - 1 + seq_feature_num, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        #self.fc = BasicFullyConnection.BasicFullyConnection(feature_num - 1 + seq_feature_num)

    def forward(self, x):
        pre_day_info = None
        seq_day_info = None
        assert x.shape[1] == self.seq_len
        # remove the last day's delay info (that's mask info)
        # pre_day_info (batch_size, feature_num - 1)
        pre_day_info = x[:, -1, :-1]
        pre_day_info = pre_day_info.reshape(-1, self.feature_num - 1)
        # seq_day_info (batch_size, seq_len - 1, feature_num)
        seq_day_info = x[:, :-1, :]
        seq_feature = self.lstm(seq_day_info)
        # join the pre_day_info and seq_feature together
        # pre_feature (batch_size, feature_num - 1 + seq_feature_num)
        pre_feature = torch.cat((pre_day_info, seq_feature), 1)
        out = self.fc(pre_feature)
        return out
