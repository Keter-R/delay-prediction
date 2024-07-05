import torch
import yaml
from torch import nn
import pytorch_lightning as pl
import DataLoader
import ModelModule
from GCN_Ver.models import GCN

# load config from config.yaml
config = yaml.load(open('./GCN_Ver/config.yaml', 'r'), Loader=yaml.FullLoader)
seed_id = 1
print(config)
pl.seed_everything(seed=config['random_seed'][seed_id], workers=True)
data = DataLoader.DataModule(data_path=config['data_path'],
                             raw_cols=config['raw_cols'],
                             time_limit=config['time_limit'],
                             graph_feature_cols=config['graph_feature_cols'], node_num=config['node_num'],
                             delt_t=config['time_interval_minutes'], split_ratio=config['split_ratio'],
                             seed=config['random_seed'][seed_id], k_neighbors=config['k_neighbors'])

gcn_model = GCN.M_GCN(node_num=data.data_len, graph_feature_num=data.graph_feature_num,
                      gcn_hidden_size=512, gcn_feature_num=512, fc_hidden_size=256,
                      using_spatial=True, using_temporal=True, using_neighbor=False).to('cuda')
# std_gcn_model = GCN.STD_M_GCN(node_num=data.data_len, graph_feature_num=data.graph_feature_num,
#                               gcn_feature_num=512, fc_hidden_size=256,
#                               using_spatial=True, using_temporal=True, using_neighbor=False).to('cuda')

print(data.np_ratio)
task = ModelModule.ModelModule(model=gcn_model,
                               loss_function=nn.BCEWithLogitsLoss(
                                   pos_weight=torch.Tensor([data.np_ratio * 1.0]).to('cuda')),
                               feature_num=data.feature_num,
                               spatial_node_num=data.data_len, spatial_feature_num=data.graph_feature_num,
                               temporal_node_num=data.data_len, temporal_feature_num=data.graph_feature_num,
                               neighbor_node_num=data.data_len, neighbor_feature_num=data.graph_feature_num,
                               lr=config['lr'], weight_decay=config['weight_decay'], lr_gamma=config['lr_gamma'])
# task_std = ModelModule.ModelModule(model=std_gcn_model,
#                                    loss_function=nn.BCEWithLogitsLoss(
#                                        pos_weight=torch.Tensor([data.np_ratio * 1.0]).to('cuda')),
#                                    feature_num=data.feature_num,
#                                    spatial_node_num=data.data_len, spatial_feature_num=data.graph_feature_num,
#                                    temporal_node_num=data.data_len, temporal_feature_num=data.graph_feature_num,
#                                    neighbor_node_num=data.data_len, neighbor_feature_num=data.graph_feature_num,
#                                    lr=config['lr'], weight_decay=config['weight_decay'], lr_gamma=config['lr_gamma'])

trainer = pl.Trainer(accelerator="gpu", devices="1", max_epochs=config['epoch'], deterministic="warn")
trainer.fit(task, datamodule=data)
result = trainer.validate(ckpt_path="best", datamodule=data)

# trainer.fit(task_std, datamodule=data)
# result_std = trainer.validate(ckpt_path="best", datamodule=data)
