import os

import torch
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
import torchmetrics as tm
from torcheval.metrics.aggregation.auc import AUC


def calculate_metrics(y_hat: Tensor, y: Tensor) -> dict:
    print(f"\ny_hat_max: {y_hat.max()}, y_hat_min: {y_hat.min()}")
    print(f"y_max: {y.max()}, y_min: {y.min()}")
    # y_hat = (y_hat > 30).float()
    # y = (y > 30).float()
    acc = tm.classification.BinaryAccuracy().to("cuda")
    recall = tm.Recall(task="binary").to("cuda")
    spec = tm.classification.BinarySpecificity().to("cuda")
    auc = tm.classification.BinaryAUROC().to("cuda")
    # value >= 0.5 as positive
    pred_positives = torch.sum(y_hat >= 0.5).item()
    pred_negatives = torch.sum(y_hat < 0.5).item()
    ground_positives = torch.sum(y).item()
    ground_negatives = len(y) - ground_positives
    # print(f"y_hat: {y_hat}")
    # print(f"y: {y}")
    print(f"\npred_positives: {pred_positives}, pred_negatives: {pred_negatives}")
    print(f"ground_positives: {ground_positives}, ground_negatives: {ground_negatives}")
    Accuracy = acc(y_hat, y).item()
    Sensitivity = 0 #recall(y_hat, y)
    Specificity = spec(y_hat, y).item()
    AUC = auc(y_hat, y).item()
    print("Accuracy: ", Accuracy)
    return {"Accuracy": Accuracy, "Sensitivity": Sensitivity, "Specificity": Specificity, "AUC": AUC}


class ModelModule(pl.LightningModule):
    def __init__(self, model, seq_len, pre_len, batch_size, loss_function, max_delay=0, lr=0.001, weight_decay=1e-4):
        super(ModelModule, self).__init__()
        self.model = model
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.max_delay = max_delay
        self.lr = lr
        self.weight_decay = weight_decay

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
        assert torch.isnan(y_hat).sum() == 0, print(f"loss: {loss} , y_hat: {y_hat}, y: {y}")
        assert torch.isnan(loss).sum() == 0, print(f"loss: {loss} , y_hat: {y_hat}, y: {y}")
        # print(y_hat)
        # exit if loss == nan
        # assert loss.item() != loss.item()
        # calculate_metrics(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.reshape((-1, 1))
        y_hat = y_hat.reshape((-1, 1))
        loss = self.loss(y_hat, y)
        metrics = calculate_metrics(y_hat.clone(), y.clone())
        self.log_dict(metrics)
        return loss

    def configure_optimizers(self):
        # optimizer = optim.Adam(
        #     self.parameters(),
        #     lr=self.lr,
        #     weight_decay=self.weight_decay,
        # )
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=0.7
        )

        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        # return optimizer

