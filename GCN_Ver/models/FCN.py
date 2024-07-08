import numpy as np
import yaml
from torch import nn
import torch
from torch_geometric.nn import GCNConv


class FCN(nn.Module):
    def __init__(self, node_num, graph_feature_num, fc_hidden_size=512):
        super(FCN, self).__init__()
        self.node_num = node_num
        self.fc = nn.Sequential(
            nn.BatchNorm1d(graph_feature_num),
            nn.Linear(graph_feature_num, fc_hidden_size),
            nn.BatchNorm1d(fc_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(fc_hidden_size, 1)
        )

    def forward(self, adj, feat):
        x = self.fc(feat[0].reshape(self.node_num, -1))
        return x
