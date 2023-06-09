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

        assert norm_layer["name"] in ["instance", "batch", "none"]

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
        encoder_layers: list,
        decoder_layers: list,
        class_layers: list,
        class_out_channels: int,
        activation: dict,
        norm_layer: dict,
        dropout: float,
        optimizer: dict,
        loss: dict,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.train_loss = torchmetrics.MeanMetric()

        self.encoder = MLP(
            in_channels=in_channels,
            hidden_layers=encoder_layers,
            activation=activation,
            norm_layer=norm_layer,
            dropout=dropout,
        )

        self.decoder = MLP(
            in_channels=encoder_layers[-1],
            hidden_layers=decoder_layers,
            activation=activation,
            norm_layer=norm_layer,
            dropout=dropout,
        )
        self.decoder = nn.Sequential(
            self.decoder, nn.Linear(decoder_layers[-1], in_channels)
        )

        self.classifier = MLP(
            in_channels=encoder_layers[-1],
            hidden_layers=class_layers,
            activation=activation,
            norm_layer=norm_layer,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            self.classifier, nn.Linear(class_layers[-1], class_out_channels)
        )

    def forward(self, x):
        latent = self.encoder(x)
        decoder_output = self.decoder(latent)

        class_output = self.classifier(latent)

        return latent, decoder_output, class_output

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)

        latent, decoder_o, class_o = self(x)

        loss = 0.0

        class_weight = self.hparams.loss["class_weight"]
        reconst_weight = self.hparams.loss["reconst_weight"]
        abs_weight = self.hparams.loss["abs_weight"]

        if class_weight > 0:
            loss += F.cross_entropy(class_o, y)

        if reconst_weight > 0:
            loss += F.mse_loss(decoder_o, x, reduction="none").sum(dim=1).mean(dim=0)

        if abs_weight > 0:
            loss += torch.abs(latent).sum(dim=1).mean(dim=0)

        self.train_loss(loss, x.size(0))

        return loss

    def on_train_epoch_end(self):
        loss = self.train_loss.compute()

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        self.train_loss.reset()

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)

        return self.encoder(x)

    def configure_optimizers(self):
        opt_cfg = self.hparams.optimizer

        opt_class = getattr(torch.optim, opt_cfg["name"])
        optimizer = opt_class(
            self.parameters(),
            **opt_cfg["init_args"],
        )
        return optimizer
