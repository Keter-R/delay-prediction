import os
import random

import numpy as np
from torch import nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from DataLoader import DataModule
import torch
import models.LSTM as LSTM
import models.BasicFullyConnection as BasicFullyConnection
import pytorch_lightning as pl
from ModelModule import ModelModule
import models.FCLstm as FCLstm
import models.GCN as GCN
import models.STGCN as STGCN
import FocalLoss
from RandomForest import RandomForest

seed = 998244353
pl.seed_everything(seed, workers=True)
torch.set_float32_matmul_precision('high')
name = 'ttc-streetcar-delay-data'
year = '2020-with-stations'
val_year = 2020
batch_size = 32
epoch = 100
seq_len = 16
pre_len = 1
hidden_size = 512
num_layers = 2

data_with_graph = DataModule(name, year, batch_size, seq_len=seq_len, label_encode=True, split_ratio=0.8,
                             with_station_id=True, graph=True)
data_with_station = DataModule(name, year, batch_size, seq_len=seq_len, label_encode=True, split_ratio=0.8,
                               with_station_id=True)
feature_num = data_with_station.feature_num


print(f"graph feature num: {data_with_graph.graph_feature_num}, seq len: {seq_len}, feature num: {feature_num}")
# rf = RandomForest(data_with_graph, seq_len, batch_size, data_with_graph.feature_num)
# rf.fit()
# rf.validate()
# exit(11111)
# model_STGCN = STGCN.STGCN(node_num=data_with_graph.node_num, feature_num=data_with_graph.feature_num
#                           , seq_len=seq_len, adj_mat=data_with_graph.adj_mat, graph_feature_num=data_with_graph.graph_feature_num)
# task_STGCN = ModelModule(model_STGCN, seq_len, pre_len, batch_size, nn.BCELoss(), max_delay=0, lr=0.001)
#
# model1 = FCLstm.FCLstm(feature_num, seq_len, seq_feature_num=256, hidden_size=hidden_size)
# task1 = ModelModule(model1, seq_len, pre_len, batch_size, nn.BCELoss(), max_delay=0, lr=0.001)

#

# task_GCN = ModelModule(model_GCN, seq_len, pre_len, batch_size, nn.BCELoss(), max_delay=0, lr=0.001)
# model_GCN2 = GCN.GCN2(node_num=data_with_station.node_num, feature_num=data_with_station.feature_num
#                       , hidden_size=hidden_size, seq_len=seq_len, adj_mat=data_with_station.adj_mat)
# task_GCN2 = ModelModule(model_GCN2, seq_len, pre_len, batch_size, nn.BCELoss(), max_delay=0, lr=0.001)
# model = BasicFullyConnection.BasicFullyConnection(data_with_station.feature_num - 1, hidden_size=hidden_size)
# task = ModelModule(model, seq_len, pre_len, batch_size, nn.BCELoss(), max_delay=0, lr=0.001)


def validate(data, name=""):
    model = ModelModule.load_from_checkpoint(checkpoint_path=f"checkpoints/{name}.ckpt")
    trainer = pl.Trainer(accelerator="gpu", devices="1")
    trainer.validate(model=model, datamodule=data)


def run(model, data, name=""):
    # loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.1]))
    ckpt_path = f"checkpoints/{name}.ckpt"
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([data.np_ratio * 1.05]).to('cuda'))
    # loss = nn.BCEWithLogitsLoss()
    task = ModelModule(model, seq_len, pre_len, batch_size, loss, max_delay=0, lr=0.001)
    trainer = pl.Trainer(accelerator="gpu", devices="1", max_epochs=epoch, deterministic="warn")
    trainer.fit(task, datamodule=data)
    result = trainer.validate(ckpt_path="best", datamodule=data)
    return result


if __name__ == '__main__':
    model = STGCN.STGCN(node_num=data_with_graph.node_num, seq_len=seq_len,
                        feature_num=data_with_graph.feature_num,
                        adj_mat=data_with_graph.adj_mat, graph_feature_num=data_with_graph.graph_feature_num,
                        fc_hidden_size=64, gcn_output_size=32)
    # model = GCN.GCN(node_num=data_with_station.node_num, feature_num=data_with_station.feature_num,
    # fc_hidden_size=16, gcn_hidden_size=4,
    # seq_len=seq_len, adj_mat=data_with_station.adj_mat)
    # model = BasicFullyConnection.FCForGraph(data_with_graph.feature_num, hidden_size=16)
    run(model, data_with_graph, "STGCN")
