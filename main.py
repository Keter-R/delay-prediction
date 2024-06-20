from torch import nn

from DataLoader import DataModule
import torch
import models.LSTM as LSTM
import models.BasicFullyConnection as BasicFullyConnection
import pytorch_lightning as pl
from ModelModule import ModelModule
import models.FCLstm as FCLstm
name = 'ttc-streetcar-delay-data'
year = '2020-with-stations'
val_year = 2020
batch_size = 32
epoch = 50
seq_len = 6
pre_len = 1
hidden_size = 512
num_layers = 2
# 0.9078455567359924

# model = LSTM.LSTM(feature_num, hidden_size, num_layers, pre_len, seq_len=seq_len)
data = DataModule(name, year, batch_size, seq_len=seq_len, pre_len=pre_len, label_encode=True, split_ratio=0.6)
# val_data = DataModule(name, val_year, batch_size, seq_len=seq_len, pre_len=pre_len, label_encode=True, split_ratio=0.6)
feature_num = data.feature_num
model = BasicFullyConnection.BasicFullyConnection(feature_num, hidden_size=hidden_size)
task = ModelModule(model, seq_len, pre_len, batch_size, nn.BCELoss(), max_delay=0, lr=0.01)
model1 = FCLstm.FCLstm(feature_num + 1, seq_len, seq_feature_num=256, hidden_size=hidden_size)
task1 = ModelModule(model, seq_len, pre_len, batch_size, nn.BCELoss(), max_delay=0, lr=0.01)
torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    trainer1 = pl.Trainer(accelerator="gpu", devices="1", max_epochs=epoch)
    trainer1.fit(task, data)
    trainer2 = pl.Trainer(accelerator="gpu", devices="1", max_epochs=epoch)
    trainer2.fit(task1, data)
    print("------BasicFullyConnection Validation----")
    result1 = trainer1.validate(ckpt_path="best", datamodule=data)
    print("------FCLstm Validation----")
    result2 = trainer2.validate(ckpt_path="best", datamodule=data)
    print("------BasicFullyConnection Result----")
    print(result1)
    print("------FCLstm Result----")
    print(result2)
