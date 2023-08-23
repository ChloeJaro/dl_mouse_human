import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics

NORM_LAYER_DICT = {
    "instance": nn.InstanceNorm1d,
    "batch": nn.BatchNorm1d,
    "none": nn.Identity,
}


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_layers: list,
        activation: dict,
        norm_layer: dict,
        dropout: float,
    ):
        super().__init__()

        assert norm_layer["name"] in NORM_LAYER_DICT

        activation_clz = getattr(torch.nn, activation["name"])
        activation_clz = functools.partial(activation_clz, **activation["init_args"])

        norm_layer_clz = NORM_LAYER_DICT[norm_layer["name"]]
        norm_layer_clz = functools.partial(norm_layer_clz, **norm_layer["init_args"])

        self.mlp = []

        prev_channels = in_channels

        for ch in hidden_layers:
            self.mlp.extend(
                [
                    nn.Linear(prev_channels, ch),
                    norm_layer_clz(ch),
                    activation_clz(),
                    nn.Dropout(p=dropout),
                ]
            )

            prev_channels = ch

        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        return self.mlp(x)


class LitNet(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        hidden_layers: list,
        out_channels: int,
        activation: dict,
        norm_layer: dict,
        dropout: float,
        optimizer: dict,
        onecycle_lr_scheduler: dict,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.automatic_optimization = onecycle_lr_scheduler is None

        self.train_loss = torchmetrics.MeanMetric()
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=out_channels, average="micro"
        )
        self.valid_loss = torchmetrics.MeanMetric()
        self.valid_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=out_channels, average="micro"
        )
			
		
        self.mlp = MLP(
            in_channels=in_channels,
            hidden_layers=hidden_layers,
            activation=activation,
            norm_layer=norm_layer,
            dropout=dropout,
        )

        self.output = nn.Linear(hidden_layers[-1], out_channels)

    def forward(self, x):
        h = self.mlp(x)
        return self.output(h)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)

        o = self(x)

        loss = F.cross_entropy(o, y)

        self.train_loss(loss, x.size(0))
        self.train_acc(o, y)

        if self.automatic_optimization:
            return loss

        opt = self.optimizers()
        sch = self.lr_schedulers()

        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        if self.trainer.is_last_batch:
            sch.step()

    def on_train_epoch_end(self):
        loss = self.train_loss.compute()
        train_acc = self.train_acc.compute()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", train_acc, prog_bar=True)

        self.train_loss.reset()
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)

        val_o = self(x)

        val_loss = F.cross_entropy(val_o, y)

        self.valid_loss(val_loss, x.size(0))
        self.valid_acc(val_o, y)

        if self.automatic_optimization:
            return loss

        opt = self.optimizers()
        sch = self.lr_schedulers()

        opt.zero_grad()
        self.manual_backward(val_loss)
        opt.step()

        if self.trainer.is_last_batch:
            sch.step()

	
    def on_validation_epoch_end(self):
        val_loss = self.valid_loss.compute()
        val_acc = self.valid_acc.compute()

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)

        self.train_loss.reset()
        self.train_acc.reset()

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)

        return self.mlp(x)

    def configure_optimizers(self):
        opt_cfg = self.hparams.optimizer

        opt_class = getattr(torch.optim, opt_cfg["name"])
        optimizer = opt_class(
            self.parameters(),
            **opt_cfg["init_args"],
        )

        optimizer_dict = {
            "optimizer": optimizer,
        }

        if self.automatic_optimization:
            return optimizer_dict

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, total_steps=self.trainer.max_epochs,
            **self.hparams.onecycle_lr_scheduler
        )

        optimizer_dict["lr_scheduler"] = lr_scheduler
        return optimizer_dict
