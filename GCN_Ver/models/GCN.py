import numpy as np
import yaml
from torch import nn
import torch
from torch_geometric.nn import GCNConv


def adj_to_edge_index(adj):
    if adj.shape[0] != adj.shape[1]:
        n = max(adj.shape[0], adj.shape[1])
        adj = adj.reshape(n, n)
    adj = np.array(adj.cpu().detach())
    # edge_index in shape of [2, E]
    # x_i -> x_j means [0, k] = i and [1, k] = j
    edge_index = []
    n = adj.shape[0]
    for i in range(0, n):
        for j in range(0, n):
            if adj[i][j] > 0:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to('cuda')
    return edge_index


class STD_M_GCN(nn.Module):
    def __init__(self, node_num, graph_feature_num, fc_hidden_size=512, gcn_feature_num=32,
                 using_spatial=True, using_temporal=False, using_neighbor=False):
        super(STD_M_GCN, self).__init__()
        self.node_num = node_num
        self.graph_feature_num = graph_feature_num
        self.using_spatial = using_spatial
        self.using_temporal = using_temporal
        self.using_neighbor = using_neighbor
        self.model_spatial = None
        self.model_temporal = None
        self.model_neighbor = None
        if using_spatial:
            self.model_spatial_conv1 = GCNConv(graph_feature_num, gcn_feature_num, improved=True)
            self.model_spatial_conv2 = GCNConv(gcn_feature_num, gcn_feature_num, improved=True)
            self.model_spatial_conv3 = GCNConv(gcn_feature_num, gcn_feature_num, improved=True)
        if using_temporal:
            self.model_temporal_conv1 = GCNConv(graph_feature_num, gcn_feature_num, improved=True)
            self.model_temporal_conv2 = GCNConv(gcn_feature_num, gcn_feature_num, improved=True)
            self.model_temporal_conv3 = GCNConv(gcn_feature_num, gcn_feature_num, improved=True)
        if using_neighbor:
            self.model_neighbor = GCNConv(graph_feature_num, gcn_feature_num, improved=True)
        gcn_output_size = gcn_feature_num * (using_spatial + using_temporal + using_neighbor)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(gcn_output_size),
            nn.Linear(gcn_output_size, fc_hidden_size),
            nn.BatchNorm1d(fc_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(fc_hidden_size, 1)
        )

    def forward(self, adj_list, feat_list):
        x_spatial = torch.zeros(0).to('cuda')
        x_temporal = torch.zeros(0).to('cuda')
        x_neighbor = torch.zeros(0).to('cuda')
        if self.using_spatial and adj_list[0] is not None and feat_list[0] is not None:
            adj = adj_list[0].reshape(self.node_num, self.node_num)
            adj = torch.tensor(adj, dtype=torch.float32).to('cuda')
            adj = adj.to_sparse()
            feat = feat_list[0].reshape(self.node_num, self.graph_feature_num)
            x_spatial = self.model_spatial_conv1(feat, adj)
            x_spatial = torch.nn.functional.relu(x_spatial)
            x_spatial = self.model_spatial_conv2(x_spatial, adj)
            x_spatial = torch.nn.functional.relu(x_spatial)
            x_spatial = self.model_spatial_conv3(x_spatial, adj)
        if self.using_temporal and adj_list[1] is not None and feat_list[1] is not None:
            adj = adj_list[1].reshape(self.node_num, self.node_num)
            adj = torch.tensor(adj, dtype=torch.float32).to('cuda')
            adj = adj.to_sparse()
            feat = feat_list[1].reshape(self.node_num, self.graph_feature_num)
            x_temporal = self.model_temporal_conv1(feat, adj)
            x_temporal = torch.nn.functional.relu(x_temporal)
            x_temporal = self.model_temporal_conv2(x_temporal, adj)
            x_temporal = torch.nn.functional.relu(x_temporal)
            x_temporal = self.model_temporal_conv3(x_temporal, adj)
        if self.using_neighbor and adj_list[2] is not None and feat_list[2] is not None:
            adj = adj_list[2].reshape(self.node_num, self.node_num)
            adj = torch.tensor(adj, dtype=torch.float32).to('cuda')
            adj = adj.to_sparse()
            feat = feat_list[2].reshape(self.node_num, self.graph_feature_num)
            x_neighbor = self.model_neighbor(feat, adj)
        x = torch.cat((x_spatial, x_temporal, x_neighbor), 1)
        x = self.fc(x)
        return x


class M_GCN(nn.Module):
    def __init__(self, node_num, graph_feature_num, gcn_hidden_size=512, fc_hidden_size=512, gcn_feature_num=32,
                 using_spatial=True, using_temporal=False, using_neighbor=False):
        super(M_GCN, self).__init__()
        self.node_num = node_num
        self.graph_feature_num = graph_feature_num
        self.gcn_hidden_size = gcn_hidden_size
        self.using_spatial = using_spatial
        self.using_temporal = using_temporal
        self.using_neighbor = using_neighbor
        self.model_spatial = None
        self.model_temporal = None
        self.model_neighbor = None
        config = yaml.load(open('./GCN_Ver/config.yaml', 'r'), Loader=yaml.FullLoader)['self_enhance']
        self.lmb = [config['spatial_adj_lmb'], config['temporal_adj_lmb'], config['neighbor_adj_lmb']]
        if using_spatial:
            self.model_spatial = GCN(node_num, graph_feature_num, gcn_hidden_size, fc_hidden_size, gcn_feature_num).to('cuda')
        if using_temporal:
            self.model_temporal = GCN(node_num, graph_feature_num, gcn_hidden_size, fc_hidden_size, gcn_feature_num).to('cuda')
        if using_neighbor:
            self.model_neighbor = GCN(node_num, graph_feature_num, gcn_hidden_size, fc_hidden_size, gcn_feature_num).to('cuda')
        gcn_out_dim = gcn_feature_num * (using_spatial + using_temporal + using_neighbor)
        # self.fc0 = nn.Sequential(
        #     nn.BatchNorm1d(graph_feature_num),
        #     # nn.Linear(graph_feature_num, fc_hidden_size),
        #     # nn.BatchNorm1d(fc_hidden_size),
        #     # nn.LeakyReLU(),
        #     # nn.Linear(fc_hidden_size, 1)
        # )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(gcn_out_dim),
            nn.Linear(gcn_out_dim, fc_hidden_size),
            nn.BatchNorm1d(fc_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(fc_hidden_size, 1)
        )

    def forward(self, adj_list, feat_list):
        x_spatial = torch.zeros(0).to('cuda')
        x_temporal = torch.zeros(0).to('cuda')
        x_neighbor = torch.zeros(0).to('cuda')
        if self.using_spatial and adj_list[0] is not None and feat_list[0] is not None:
            x_spatial = self.model_spatial(adj_list[0], feat_list[0], self.lmb[0]).to('cuda')
        if self.using_temporal and adj_list[1] is not None and feat_list[1] is not None:
            x_temporal = self.model_temporal(adj_list[1], feat_list[1], self.lmb[1]).to('cuda')
        if self.using_neighbor and adj_list[2] is not None and feat_list[2] is not None:
            x_neighbor = self.model_neighbor(adj_list[2], feat_list[2], self.lmb[2]).to('cuda')
        # _x = self.fc0(feat_list[0].reshape(self.node_num, self.graph_feature_num))
        x = torch.cat((x_spatial, x_temporal, x_neighbor), 1)
        # x = torch.cat((x, _x), 1)
        x = self.fc(x)
        return x


class GCN(nn.Module):
    def __init__(self, node_num, graph_feature_num, gcn_hidden_size=512, fc_hidden_size=512, out_feature_num=1):
        super(GCN, self).__init__()
        self.node_num = node_num
        self.gcn_hidden_size = gcn_hidden_size
        self.out_feature_num = out_feature_num
        self.gc1 = nn.Linear(graph_feature_num, gcn_hidden_size, bias=True)
        self.gc2 = nn.Linear(gcn_hidden_size, out_feature_num, bias=True)
        # self.gc3 = nn.Linear(gcn_hidden_size // 2, gcn_hidden_size, bias=True)
        self.A = None
        self.training = True
        self.fc = nn.Sequential(
            nn.BatchNorm1d(gcn_hidden_size),
            nn.Linear(gcn_hidden_size, fc_hidden_size),
            nn.BatchNorm1d(fc_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(fc_hidden_size, out_feature_num)
        )

    def forward(self, adj, feat, lmb):
        # prepare A for GCN
        if self.A is None:
            adj = adj.reshape(self.node_num, self.node_num)
            A = torch.tensor(adj, dtype=torch.float32).to('cuda') + torch.eye(self.node_num).to('cuda')*lmb
            D = torch.diag(torch.sum(A, dim=1))
            D = torch.inverse(torch.sqrt(D))
            A = torch.mm(torch.mm(D, A), D)
            self.A = A
        else:
            A = self.A
        X = feat.reshape(self.node_num, -1)
        X = torch.nn.functional.relu(self.gc1(A @ X))
        X = torch.nn.functional.relu(self.gc2(A @ X))
        # X = self.gc3(A @ X)
        # X = torch.nn.functional.relu(self.gc3(A @ X))
        # X = self.fc(X)
        X = X.reshape(self.node_num, self.out_feature_num)
        return X
