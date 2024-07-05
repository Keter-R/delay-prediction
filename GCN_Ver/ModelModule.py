import os

import torch
import torcheval
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
import torchmetrics as tm
from torcheval.metrics.aggregation.auc import AUC


def calculate_metrics(y_hat: Tensor, y: Tensor) -> dict:
    # if y_hat out of range [0, 1]
    # apply sigmoid function to y_hat
    if y_hat.max() > 1 or y_hat.min() < 0:
        y_hat = torch.sigmoid(y_hat)
    auc = tm.classification.BinaryAUROC().to("cuda")
    acc = tm.classification.BinaryAccuracy().to("cuda")
    recall = tm.classification.BinaryRecall().to("cuda")
    spec = tm.classification.BinarySpecificity().to("cuda")
    auc = auc(y_hat, y).item()
    Accuracy = acc(y_hat, y).item()
    Sensitivity = recall(y_hat, y).item()
    Specificity = spec(y_hat, y).item()
    y_hat = (y_hat >= 0.5).float()
    TP = (y_hat * y).sum().item()
    TN = ((1 - y_hat) * (1 - y)).sum().item()
    FP = (y_hat * (1 - y)).sum().item()
    FN = ((1 - y_hat) * y).sum().item()
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    return {"Accuracy": Accuracy, "AUC": auc, "Sensitivity": Sensitivity, "Specificity": Specificity,
            "FPR": FPR, "FNR": FNR, "TP": TP, "TN": TN, "FP": FP, "FN": FN}


class ModelModule(pl.LightningModule):
    def __init__(self, model, loss_function,
                 feature_num=0,
                 spatial_node_num=0, spatial_feature_num=0,
                 temporal_node_num=0, temporal_feature_num=0,
                 neighbor_node_num=0, neighbor_feature_num=0,
                 lr=0.001, weight_decay=1e-3,
                 lr_gamma=0.83):
        super(ModelModule, self).__init__()
        self.save_hyperparameters()
        self.model = model
        self.loss_function = loss_function
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.train_step_outputs = []
        self.train_step_targets = []
        self.val_step_outputs = []
        self.val_step_targets = []
        self.spatial_node_num = spatial_node_num
        self.spatial_feature_num = spatial_feature_num
        self.temporal_node_num = temporal_node_num
        self.temporal_feature_num = temporal_feature_num
        self.neighbor_node_num = neighbor_node_num
        self.neighbor_feature_num = neighbor_feature_num
        self.feature_num = feature_num

    def forward(self, x, y):
        adj = [None, None, None]
        feat = [None, None, None]
        pre_index = 0
        if self.spatial_node_num > 0:
            adj[0] = x[:, :self.spatial_node_num ** 2].reshape(1, self.spatial_node_num, self.spatial_node_num)
            feat[0] = x[:, self.spatial_node_num ** 2: self.spatial_node_num ** 2 + self.spatial_node_num * self.spatial_feature_num].reshape(1, self.spatial_node_num, self.spatial_feature_num)
            pre_index = self.spatial_node_num ** 2 + self.spatial_node_num * self.spatial_feature_num
        if self.temporal_node_num > 0:
            adj[1] = x[:, pre_index: pre_index + self.temporal_node_num ** 2].reshape(1, self.temporal_node_num, self.temporal_node_num)
            feat[1] = x[:, pre_index + self.temporal_node_num ** 2: pre_index + self.temporal_node_num ** 2 + self.temporal_node_num * self.temporal_feature_num].reshape(1, self.temporal_node_num, self.temporal_feature_num)
            pre_index = pre_index + self.temporal_node_num ** 2 + self.temporal_node_num * self.temporal_feature_num
        if self.neighbor_node_num > 0:
            adj[2] = x[:, pre_index: pre_index + self.neighbor_node_num ** 2].reshape(1, self.neighbor_node_num, self.neighbor_node_num)
            feat[2] = x[:, pre_index + self.neighbor_node_num ** 2: pre_index + self.neighbor_node_num ** 2 + self.neighbor_node_num * self.neighbor_feature_num].reshape(1, self.neighbor_node_num, self.neighbor_feature_num)
        pred = self.model(adj, feat)
        pred = pred.flatten()
        y_index = y[0, 0, :].long()
        y = y[0, 1, :]
        pred = pred[y_index]
        return pred, y

    def loss(self, y_hat, y):
        if y.device != 'cuda':
            y = y.to('cuda')
        if y_hat.device != 'cuda':
            y_hat = y_hat.to('cuda')
        return self.loss_function(y_hat, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, y = self(x, y)
        loss = self.loss(y_hat, y)
        self.train_step_outputs.extend(y_hat.clone().cpu().tolist())
        self.train_step_targets.extend(y.clone().cpu().tolist())
        return loss

    def on_train_epoch_end(self):
        train_all_outputs = torch.Tensor(self.train_step_outputs).reshape((-1, 1))
        train_all_targets = torch.Tensor(self.train_step_targets).reshape((-1, 1))
        with torch.no_grad():
            loss = self.loss(train_all_outputs, train_all_targets)
            metrics = dict()
            metrics["train_loss"] = loss
            metrics["step"] = self.current_epoch
            self.log_dict(metrics)
        self.train_step_outputs.clear()
        self.train_step_targets.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, y = self(x, y)
        loss = self.loss(y_hat, y)
        self.val_step_outputs.extend(y_hat.clone().cpu().tolist())
        self.val_step_targets.extend(y.clone().cpu().tolist())
        return loss

    def on_validation_epoch_end(self):
        val_all_outputs = torch.Tensor(self.val_step_outputs).reshape((-1, 1))
        val_all_targets = torch.Tensor(self.val_step_targets).reshape((-1, 1))
        loss = self.loss(val_all_outputs, val_all_targets)
        metrics = calculate_metrics(val_all_outputs, val_all_targets)
        metrics["step"] = self.current_epoch
        metrics["val_loss"] = loss
        self.log_dict(metrics)
        print(metrics)
        self.val_step_outputs.clear()
        self.val_step_targets.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_gamma)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
