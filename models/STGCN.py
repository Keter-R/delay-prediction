import torch
from torch import nn
import models.TPModels.GCN.STMGCN as STMGCN
from models.TPModels.GCN.GCN import Adj_Preprocessor

class STGCN(nn.Module):
    def __init__(self, feature_num, seq_len, node_num, adj_mat, hidden_size=64, fc_hidden_size=512, gcn_output_size=512
                 , graph_feature_num=1):
        super(STGCN, self).__init__()
        self.feature_num = feature_num
        self.seq_len = seq_len
        self.node_num = node_num
        self.adj_mat = adj_mat
        sta_kernel_config = {'kernel_type': 'chebyshev', 'K': 2}
        adj_preprocessor = Adj_Preprocessor(kernel_type=sta_kernel_config['kernel_type'], K=sta_kernel_config['K'])
        self.ADJ = adj_preprocessor.process(torch.tensor(self.adj_mat, dtype=torch.float32))
        self.ADJ = self.ADJ.to('cuda')
        self.ADJ = [self.ADJ]
        self.graph_feature_num = graph_feature_num
        #self.stmgcn_feature_num = feature_num
        self.stmgcn_output_dim = gcn_output_size
        self.stmgcn_feature_num = graph_feature_num
        # self.stmgcn_output_dim = gcn_hidden_size
        self.stmgcn = STMGCN.ST_MGCN(M=1, seq_len=seq_len, n_nodes=node_num, input_dim=self.stmgcn_feature_num, lstm_hidden_dim=64, lstm_num_layers=3,
                               gcn_hidden_dim=64, sta_kernel_config=sta_kernel_config, gconv_use_bias=True, gconv_activation=nn.ReLU, output_dim=self.stmgcn_output_dim)
        self.fc0 = nn.Sequential(
            nn.Linear(feature_num - 1, fc_hidden_size),
            nn.BatchNorm1d(fc_hidden_size),
            nn.LeakyReLU()
        )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(fc_hidden_size + self.stmgcn_output_dim),
            nn.Linear(fc_hidden_size + self.stmgcn_output_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # for each batch x is flattened from (seq_len, node_num, graph_feature_num) + (1, feature_num)
        batch_size = x.shape[0]
        # extract x1 [batch_size, (seq_len) * node_num * graph_feature_num]
        # extract x2 [batch_size, feature_num]
        x1 = x[:, :-self.feature_num]
        x2 = x[:, -self.feature_num:]
        # reshape x1 to [batch_size, seq_len, node_num, graph_feature_num] aka [batch_size, T, V, P]
        x1 = x1.reshape(batch_size, self.seq_len, self.node_num, self.graph_feature_num)
        # x1 is the input of the ST_MGCN
        x1 = self.stmgcn(x1, self.ADJ)
        x2 = x2.reshape(batch_size, self.feature_num)
        sid = x2[:, -1].reshape(-1).cpu().numpy().astype(int)
        x2 = x2[:, :-1]
        gFeature = torch.zeros((x2.shape[0], self.stmgcn_output_dim))
        gFeature = gFeature.to(x2.device)
        for i in range(x2.shape[0]):
            gFeature[i, :] = x1[i, sid[i], :]
        x2 = x2.to(x.device)
        x2 = self.fc0(x2)
        gFeature = gFeature.to(x.device)
        x = torch.cat((x2, gFeature), 1)
        x = self.fc(x)
        return x