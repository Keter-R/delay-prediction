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
batch_size = 128
epoch = 50
seq_len = 16
pre_len = 1
hidden_size = 512
num_layers = 2

data_with_graph = DataModule(name, year, batch_size, seq_len=seq_len, label_encode=True, split_ratio=0.8, with_station_id=True, graph=True)
# data_with_station = DataModule(name, year, batch_size, seq_len=seq_len, label_encode=True, split_ratio=0.8, with_station_id=True)
# feature_num = data_with_station.feature_num
# # load model
name = "STGCN"
# #  checkpoint_path = "checkpoints/" + name + ".ckpt"
checkpoint_path = f"lightning_logs/{name}/checkpoints/epoch=57-step=57.ckpt"
# hyper_path = f"lightning_logs/{name}/hparams.yaml"
#
#
if __name__ == '__main__':
    model = ModelModule.load_from_checkpoint(checkpoint_path=checkpoint_path)
    trainer = pl.Trainer(accelerator="gpu", devices="1")
    trainer.validate(model=model, datamodule=data_with_graph)






