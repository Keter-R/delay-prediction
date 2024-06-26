import os
import random

import numpy as np
from torch import nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from DataLoader import DataModule
from ModelModule import calculate_metrics
import torch
import models.LSTM as LSTM
import models.BasicFullyConnection as BasicFullyConnection
import pytorch_lightning as pl
from ModelModule import ModelModule
import models.FCLstm as FCLstm
import models.GCN as GCN
import models.STGCN as STGCN
from torchvision.ops import sigmoid_focal_loss

seed = 998244353
pl.seed_everything(seed, workers=True)
# os.environ['PYTHONHASHSEED'] = str(seed)
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'    # make sure CUDA utilization is stable with CUDA Version >= 10.2
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False
# torch.use_deterministic_algorithms(True)
# torch.Generator().manual_seed(seed)
# random.seed(seed)
# np.random.seed(seed)
torch.set_float32_matmul_precision('high')

name = 'ttc-streetcar-delay-data'
year = '2020-with-stations'
val_year = 2020
batch_size = 32
epoch = 50
seq_len = 16
pre_len = 1
hidden_size = 512
num_layers = 2

data_with_graph = DataModule(name, year, batch_size, seq_len=seq_len, label_encode=True, split_ratio=0.8, with_station_id=True, graph=True)
data_with_graph.setup()
# data_with_station = DataModule(name, year, batch_size, seq_len=seq_len, label_encode=True, split_ratio=0.8, with_station_id=True)
# feature_num = data_with_station.feature_num
# # load model
name = "STGCN_512_32"
checkpoint_path = "lightning_logs/{}/checkpoints/epoch=49-step=49.ckpt".format(name)


def train():
    model = STGCN.STGCN(node_num=data_with_graph.node_num, seq_len=seq_len, feature_num=data_with_graph.feature_num,
                        adj_mat=data_with_graph.adj_mat, graph_feature_num=data_with_graph.graph_feature_num,
                        fc_hidden_size=512, gcn_output_size=32)
    model = model.to("cuda")
    for i in range(epoch):
        model.train(True)
        train_loss = 0.0
        train_batches = len(data_with_graph.train_dataloader())
        loss_fn = nn.BCELoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        for j, data in enumerate(data_with_graph.train_dataloader()):
            opt.zero_grad()
            x, y = data
            x = x.to("cuda")
            y = y.to("cuda").reshape(-1)
            y_hat = model(x).reshape(-1)
            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss = train_loss / train_batches
        print(f"epoch: {i}, train loss: {train_loss}")
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for j, data in enumerate(data_with_graph.val_dataloader()):
                x, y = data
                x = x.to("cuda")
                y = y.to("cuda").reshape(-1)
                y_hat = model(x).reshape(-1)
                loss = loss_fn(y_hat, y)
                val_loss += loss.item()
            val_loss /= len(data_with_graph.val_dataloader())
            print(f"epoch: {i}, val_loss: {val_loss}")


if __name__ == '__main__':
    train()






