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
        self.lstm = LSTM_FEAT.LSTM_FEAT(feature_num, seq_feature_num, seq_len=seq_len - 1, hidden_size=512)
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
        # assert x.shape[1] == self.seq_len
        # remove the last day's delay info (that's mask info)
        # pre_day_info (batch_size, feature_num - 1)
        pre_day_info = x[:, -1, :-2]
        pre_day_info = pre_day_info.reshape(-1, self.feature_num - 1)
        # seq_day_info (batch_size, seq_len - 1, 0-feature_num-2&feature_num-1)
        seq_day_info = torch.cat((x[:, :-1, :-2], x[:, :-1, -1].reshape(x.shape[0], -1, 1)), 2)
        seq_feature = self.lstm(seq_day_info)
        # join the pre_day_info and seq_feature together
        # pre_feature (batch_size, feature_num - 1 + seq_feature_num)
        pre_feature = torch.cat((pre_day_info, seq_feature), 1)
        out = self.fc(pre_feature)
        return out



class FCLstm_with_info_enhanced(nn.Module):
    def __init__(self, feature_num, seq_len, seq_feature_num=128, info_enhance_size=512, hidden_size=128, num_layers=3):
        super(FCLstm_with_info_enhanced, self).__init__()
        self.feature_num = feature_num
        self.seq_len = seq_len
        self.seq_feature_num = seq_feature_num
        self.info_enhance_size = info_enhance_size
        self.lstm = LSTM_FEAT.LSTM_FEAT(feature_num, seq_feature_num, seq_len=seq_len - 1, hidden_size=512)
        self.fc0 = nn.Sequential(
            nn.Linear(feature_num - 1, info_enhance_size),
            nn.BatchNorm1d(info_enhance_size),
            nn.LeakyReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(info_enhance_size + seq_feature_num, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        #self.fc = BasicFullyConnection.BasicFullyConnection(feature_num - 1 + seq_feature_num)

    def forward(self, x):
        pre_day_info = None
        seq_day_info = None
        # assert x.shape[1] == self.seq_len
        # remove the last day's delay info (that's mask info)
        # pre_day_info (batch_size, feature_num - 1)
        pre_day_info = x[:, -1, :-2]
        pre_day_info = pre_day_info.reshape(-1, self.feature_num - 1)
        pre_day_info = self.fc0(pre_day_info)
        # seq_day_info (batch_size, seq_len - 1, 0-feature_num-2&feature_num-1)
        seq_day_info = torch.cat((x[:, :-1, :-2], x[:, :-1, -1].reshape(x.shape[0], -1, 1)), 2)
        seq_feature = self.lstm(seq_day_info)
        # join the pre_day_info and seq_feature together
        # pre_feature (batch_size, feature_num - 1 + seq_feature_num)
        pre_feature = torch.cat((pre_day_info, seq_feature), 1)
        out = self.fc(pre_feature)
        return out