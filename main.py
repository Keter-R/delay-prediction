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
torch.use_deterministic_algorithms(warn_only=True, mode=True)
name = 'ttc-streetcar-delay-data'
year = '2020-with-stations'
val_year = 2020
batch_size = 32
epoch = 50
seq_len = 16
pre_len = 1
hidden_size = 512
num_layers = 2
# 0.9078455567359924

# model = LSTM.LSTM(feature_num, hidden_size, num_layers, pre_len, seq_len=seq_len)
# data = DataModule(name, year, batch_size, seq_len=seq_len, pre_len=pre_len, label_encode=True, split_ratio=0.8)
# # val_data = DataModule(name, val_year, batch_size, seq_len=seq_len, pre_len=pre_len, label_encode=True, split_ratio=0.6)
# feature_num = data.feature_num
data_with_graph = DataModule(name, year, batch_size, seq_len=seq_len, label_encode=True, split_ratio=0.8, with_station_id=True, graph=True)
data_with_station = DataModule(name, year, batch_size, seq_len=seq_len, label_encode=True, split_ratio=0.8, with_station_id=True)
feature_num = data_with_station.feature_num

# model_STGCN = STGCN.STGCN(node_num=data_with_graph.node_num, feature_num=data_with_graph.feature_num
#                           , seq_len=seq_len, adj_mat=data_with_graph.adj_mat, graph_feature_num=data_with_graph.graph_feature_num)
# task_STGCN = ModelModule(model_STGCN, seq_len, pre_len, batch_size, nn.BCELoss(), max_delay=0, lr=0.001)
#
# model1 = FCLstm.FCLstm(feature_num, seq_len, seq_feature_num=256, hidden_size=hidden_size)
# task1 = ModelModule(model1, seq_len, pre_len, batch_size, nn.BCELoss(), max_delay=0, lr=0.001)

#
# model_GCN = GCN.GCN(node_num=data_with_station.node_num, feature_num=data_with_station.feature_num
#                     , hidden_size=hidden_size, seq_len=seq_len, adj_mat=data_with_station.adj_mat)
# task_GCN = ModelModule(model_GCN, seq_len, pre_len, batch_size, nn.BCELoss(), max_delay=0, lr=0.001)
# model_GCN2 = GCN.GCN2(node_num=data_with_station.node_num, feature_num=data_with_station.feature_num
#                       , hidden_size=hidden_size, seq_len=seq_len, adj_mat=data_with_station.adj_mat)
# task_GCN2 = ModelModule(model_GCN2, seq_len, pre_len, batch_size, nn.BCELoss(), max_delay=0, lr=0.001)
# model = BasicFullyConnection.BasicFullyConnection(data_with_station.feature_num - 1, hidden_size=hidden_size)
# task = ModelModule(model, seq_len, pre_len, batch_size, nn.BCELoss(), max_delay=0, lr=0.001)

def FCTest():
    # FCM0 = BasicFullyConnection.BasicFullyConnection(data_with_station.feature_num - 1, hidden_size=hidden_size)
    # FCM2 = BasicFullyConnection.BasicFullyConnection(data_with_station.feature_num - 1, hidden_size=hidden_size)
    FCM1 = BasicFullyConnection.FCForGraph(data_with_graph.feature_num, hidden_size=hidden_size)
    # task0 = ModelModule(FCM0, seq_len, pre_len, batch_size, nn.BCELoss(), max_delay=0, lr=0.001)
    task1 = ModelModule(FCM1, seq_len, pre_len, batch_size, nn.BCELoss(), max_delay=0, lr=0.001)
    # task2 = ModelModule(FCM2, seq_len, pre_len, batch_size, nn.BCELoss(), max_delay=0, lr=0.001)
    # trainer0 = pl.Trainer(deterministic=True, accelerator="gpu", devices="1", max_epochs=epoch)
    # trainer0.fit(task0, data_with_station)
    trainer1 = pl.Trainer(accelerator="gpu", devices="1", max_epochs=epoch)
    trainer1.fit(task1, data_with_graph)
    # trainer2 = pl.Trainer(deterministic=True, accelerator="gpu", devices="1", max_epochs=epoch)
    # trainer2.fit(task2, data_with_station)
    # print("------data_with_station Validation----")
    # result0 = trainer0.validate(ckpt_path="best", datamodule=data_with_station)
    print("------data_with_graph Validation----")
    result1 = trainer1.validate(ckpt_path="best", datamodule=data_with_graph)
    # print("------data_with_station Validation----")
    # result2 = trainer2.validate(ckpt_path="best", datamodule=data_with_station)


def run(model, data, name=""):
    # loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.1]))
    loss = nn.BCELoss()
    task = ModelModule(model, seq_len, pre_len, batch_size, loss, max_delay=0, lr=0.001)
    trainer = pl.Trainer(accelerator="gpu", devices="1", max_epochs=epoch)
    trainer.fit(task, datamodule=data)
    print(f"------{name} Validation----")
    result = trainer.validate(ckpt_path="best", datamodule=data, verbose=True)
    return result


if __name__ == '__main__':
    model = STGCN.STGCN(node_num=data_with_graph.node_num, seq_len=seq_len, feature_num=data_with_graph.feature_num,
                        adj_mat=data_with_graph.adj_mat, graph_feature_num=data_with_graph.graph_feature_num,
                        fc_hidden_size=512, gcn_output_size=32)
    # model = BasicFullyConnection.FCForGraph(data_with_graph.feature_num, hidden_size=512)
    run(model, data_with_graph, "FCN")
    # trainer0 = pl.Trainer(accelerator="gpu", devices="1", max_epochs=epoch)
    # trainer0.fit(task_STGCN, data_with_graph)
    # trainer1 = pl.Trainer(accelerator="gpu", devices="1", max_epochs=epoch)
    # trainer1.fit(task, data_with_station)
    # trainer2 = pl.Trainer(accelerator="gpu", devices="1", max_epochs=epoch)
    # trainer2.fit(task1, data_with_station)
    # trainer3 = pl.Trainer(accelerator="gpu", devices="1", max_epochs=epoch)
    # trainer3.fit(task_GCN, data_with_station)
    # trainer4= pl.Trainer(accelerator="gpu", devices="1", max_epochs=epoch)
    # trainer4.fit(task_GCN2, data_with_station)
    # print("------STGCN Validation----")
    # result0 = trainer0.validate(ckpt_path="best", datamodule=data_with_graph)
    # print("------BasicFullyConnection Validation----")
    # result1 = trainer1.validate(ckpt_path="best", datamodule=data_with_station)
    # print("------FCLstm Validation----")
    # result2 = trainer2.validate(ckpt_path="best", datamodule=data_with_station)
    # print("------GCN Validation----")
    # result3 = trainer3.validate(ckpt_path="best", datamodule=data_with_station)
    # print("------GCN2 Validation----")
    # result4 = trainer4.validate(ckpt_path="best", datamodule=data_with_station)



