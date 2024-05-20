from torch import nn

from DataLoader import DataModule
import torch
import models.LSTM as LSTM
import models.BasicFullyConnection as BasicFullyConnection
import pytorch_lightning as pl
from ModelModule import ModelModule

name = 'ttc-streetcar-delay-data'
year = 2020
batch_size = 32
epoch = 300
seq_len = 1
pre_len = 1
hidden_size = 128
num_layers = 2
# 0.9078455567359924

# model = LSTM.LSTM(feature_num, hidden_size, num_layers, pre_len, seq_len=seq_len)
data = DataModule(name, year, batch_size, seq_len=seq_len, pre_len=pre_len, label_encode=True, split_ratio=0.8)
feature_num = data.feature_num
model = BasicFullyConnection.BasicFullyConnection(feature_num * seq_len)
task = ModelModule(model, seq_len, pre_len, batch_size, nn.BCELoss(), max_delay=0, lr=0.0001)
torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    trainer = pl.Trainer(accelerator="gpu", devices="1", max_epochs=epoch)
    trainer.fit(task, data)
    result = trainer.validate(datamodule=data)
    print(result)
