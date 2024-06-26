import os

import torch
import torcheval
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
import torchmetrics as tm
from torcheval.metrics.aggregation.auc import AUC

def calculate_metrics(y_hat: Tensor, y: Tensor) -> dict:
    # print(f"\ny_hat_max: {y_hat.max()}, y_hat_min: {y_hat.min()}")
    # print(f"y_max: {y.max()}, y_min: {y.min()}")
    # y_hat = (y_hat > 30).float()
    # y = (y > 30).float()
    # set values to 0 or 1
    y_hat = (y_hat >= 0.5).float()
    TP = (y_hat * y).sum().item()
    TN = ((1 - y_hat) * (1 - y)).sum().item()
    FP = (y_hat * (1 - y)).sum().item()
    FN = ((1 - y_hat) * y).sum().item()
    acc = tm.classification.BinaryAccuracy().to("cuda")
    recall = tm.Recall(task="binary").to("cuda")
    spec = tm.classification.BinarySpecificity().to("cuda")
    auc = tm.classification.BinaryAUROC().to("cuda")
    Accuracy = acc(y_hat, y).item()
    Sensitivity = recall(y_hat, y).item()
    Specificity = spec(y_hat, y).item()
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    return {"Accuracy": Accuracy, "AUC": auc(y_hat, y).item(), "Sensitivity": Sensitivity, "Specificity": Specificity,
            "FPR": FPR, "FNR": FNR, "TP": TP, "TN": TN, "FP": FP, "FN": FN}


class ModelModule(pl.LightningModule):
    def __init__(self, model, seq_len, pre_len, batch_size, loss_function, max_delay=0, lr=0.001, weight_decay=3e-5):
        super(ModelModule, self).__init__()
        self.save_hyperparameters()
        self.model = model
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.max_delay = max_delay
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_step_outputs = []
        self.train_step_targets = []
        self.val_step_outputs = []
        self.val_step_targets = []
        self.cnt = 0

    def forward(self, x):
        return self.model(x)

    def loss(self, y_hat, y):
        return self.loss_function(y_hat, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # # test
        # cnt, tnt = 0, 0
        # for i in range(len(y)):
        #     if y[i] == 1:
        #         cnt += 1
        #     else:
        #         tnt += 1
        # print(f"cnt: {cnt}, tnt: {tnt}")
        # # end test
        y_hat = self(x)
        y = y.reshape((-1, 1))
        y_hat = y_hat.reshape((-1, 1))
        loss = self.loss(y_hat, y)
        # calculate_metrics(y_hat, y)
        # assert torch.isnan(y_hat).sum() == 0, print(f"loss: {loss} , y_hat: {y_hat}, y: {y}")
        # assert torch.isnan(loss).sum() == 0, print(f"loss: {loss} , y_hat: {y_hat}, y: {y}")
        # exit if loss == nan
        # assert loss.item() != loss.item()
        # calculate_metrics(y_hat, y)
        self.log("train_loss", loss, batch_size=self.batch_size)
        # print(f"train_loss: {loss}")
        # self.train_step_outputs.extend(y_hat.argmax(dim=1).cpu().numpy())
        # self.train_step_targets.extend(y.cpu().numpy())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.reshape((-1, 1))
        y_hat = y_hat.reshape((-1, 1))
        loss = self.loss(y_hat, y)
        self.val_step_outputs.extend(y_hat.clone().cpu().tolist())
        self.val_step_targets.extend(y.clone().cpu().tolist())
        # print(f"y_hat: {self.val_step_outputs}")
        # print(f"y: {self.val_step_targets}")
        # print(len(self.val_step_targets))
        # metrics = calculate_metrics(y_hat.clone(), y.clone())
        # add loss to metrics
        # metrics["val_loss"] = loss
        # metrics["step"] = self.current_epoch
        # self.log_dict(metrics, on_step=False, on_epoch=True, batch_size=self.batch_size)
        # attach y_hat to self.val_step_outputs and y to self.val_step_targets with 1D tensor
        # print(y.shape)
        # print(f"val_metrics: {metrics}")
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
        # optimizer = optim.SGD(
        #     self.parameters(),
        #     lr=self.lr,
        #     weight_decay=self.weight_decay,
        #     momentum=0.7
        # )

        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        # return optimizer

