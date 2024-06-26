import numpy as np
import torch
from torch import nn


class GCN(nn.Module):
    def __init__(self, node_num, seq_len, feature_num, adj_mat, gcn_hidden_size=512, fc_hidden_size=512):
        super(GCN, self).__init__()
        self.gcn_hidden_size = gcn_hidden_size
        self.adj_mat = adj_mat
        self.node_num = node_num
        self.seq_len = seq_len
        self.feature_num = feature_num
        adj_mat = torch.tensor(adj_mat, dtype=torch.float32)
        A = adj_mat + torch.eye(node_num)
        D = torch.inverse(torch.sqrt(torch.diag(torch.sum(A, dim=1))))
        self.A = torch.tensor(torch.mm(torch.mm(D, A), D))
        self.gc1 = nn.Linear(feature_num, gcn_hidden_size//2, bias=True)
        self.gc2 = nn.Linear(gcn_hidden_size//2, gcn_hidden_size//2, bias=True)
        self.gc3 = nn.Linear(gcn_hidden_size//2, gcn_hidden_size, bias=True)
        self.fc0 = nn.Sequential(
            nn.Linear(feature_num - 1, fc_hidden_size),
            nn.BatchNorm1d(fc_hidden_size),
            nn.LeakyReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(gcn_hidden_size + fc_hidden_size, fc_hidden_size),
            # nn.Linear(feature_num - 1, 128),
            nn.BatchNorm1d(fc_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(fc_hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch_size, seq_len, feature_num)
        X = np.zeros((x.shape[0], self.node_num, self.feature_num))
        X = torch.tensor(X, dtype=torch.float32)
        station_ids = x[:, -1, -2].reshape(-1).cpu().numpy().astype(int)
        for i in range(x.shape[0]):
            for j in range(x.shape[1] - 1):
                sid = int(x[i, j, -2])
                feat = x[i, j, :-2]
                t = torch.squeeze(x[i, j, -1], 0).reshape(1)
                feat = torch.cat((feat, t), 0)
                X[i, sid, :] = feat
        x = x[:, -1, :-2].reshape(x.shape[0], -1)
        x = self.fc0(x)
        X = torch.tensor(X, dtype=torch.float32, device=x.device)
        self.A = self.A.to(x.device)
        # graph convolution
        X = torch.nn.functional.relu(self.gc1(self.A @ X))
        X = torch.nn.functional.relu(self.gc2(self.A @ X))
        X = torch.nn.functional.relu(self.gc3(self.A @ X))
        gFeature = np.zeros((x.shape[0], self.gcn_hidden_size))
        gFeature = torch.tensor(gFeature, dtype=torch.float32)
        for i in range(x.shape[0]):
            gFeature[i, :] = X[i, station_ids[i], :]
        gFeature = gFeature.to(x.device)
        # cat the gFeature and x to [batch_size, feature_num + hidden_size]
        x = torch.cat((x, gFeature), 1)
        x = self.fc(x)
        return x


class GCN2(nn.Module):
    def __init__(self, node_num, seq_len, feature_num, adj_mat, hidden_size=1024, fc_hidden_size=512):
        super(GCN2, self).__init__()
        self.hidden_size = hidden_size
        self.adj_mat = adj_mat
        self.node_num = node_num
        self.seq_len = seq_len
        self.feature_num = feature_num
        adj_mat = torch.tensor(adj_mat, dtype=torch.float32)
        A = adj_mat + torch.eye(node_num)
        D = torch.inverse(torch.sqrt(torch.diag(torch.sum(A, dim=1))))
        self.A = torch.tensor(torch.mm(torch.mm(D, A), D))
        self.gc1 = nn.Linear(feature_num - 1, hidden_size//2, bias=True)
        self.gc2 = nn.Linear(hidden_size//2, hidden_size//2, bias=True)
        self.gc3 = nn.Linear(hidden_size//2, hidden_size, bias=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden_size),
            # nn.Linear(feature_num - 1, 128),
            nn.BatchNorm1d(fc_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(fc_hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch_size, seq_len, feature_num)
        X = np.zeros((x.shape[0], self.node_num, self.feature_num - 1))
        X = torch.tensor(X, dtype=torch.float32)
        station_ids = x[:, -1, -2].reshape(-1).cpu().numpy().astype(int)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                sid = int(x[i, j, -2])
                X[i, sid, :] = x[i, j, :-2]
        X = torch.tensor(X, dtype=torch.float32, device=x.device)
        self.A = self.A.to(x.device)
        # graph convolution
        X = torch.nn.functional.relu(self.gc1(self.A @ X))
        X = torch.nn.functional.relu(self.gc2(self.A @ X))
        X = torch.nn.functional.relu(self.gc3(self.A @ X))
        gFeature = np.zeros((x.shape[0], self.hidden_size))
        gFeature = torch.tensor(gFeature, dtype=torch.float32)
        for i in range(x.shape[0]):
            gFeature[i, :] = X[i, station_ids[i], :]
        gFeature = gFeature.to(x.device)
        gFeature = self.fc(gFeature)
        return gFeature