import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics

from torch import Tensor

import numpy as np


NORM_LAYER_DICT = {
    "instance": nn.InstanceNorm1d,
    "batch": nn.BatchNorm1d,
    "none": nn.Identity,
}

kl_weight = 0.001

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



class LitNet_w(pl.LightningModule):
    def __init__(
        self,
        weights: list,
        latent_dim: int,
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
        onecycle_lr_scheduler: dict,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.train_loss = torchmetrics.MeanMetric()
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=class_out_channels, average="macro"
        )
        self.val_loss = torchmetrics.MeanMetric()
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=class_out_channels, average="micro"
        )

        self.w = weights
        
        self.encoder = MLP(
            in_channels=in_channels,
            hidden_layers=encoder_layers,
            activation=activation,
            norm_layer=norm_layer,
            dropout=dropout,
        )

        self.hidden2mu = nn.Linear(encoder_layers[-1], latent_dim)
        self.hidden2log_var = nn.Linear(encoder_layers[-1], latent_dim)

        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        self.decoder = MLP(
            in_channels=latent_dim,
            hidden_layers=decoder_layers,
            activation=activation,
            norm_layer=norm_layer,
            dropout=dropout,
        )

        self.decoder = nn.Sequential(
            self.decoder, nn.Linear(decoder_layers[-1], in_channels)
        )

        self.classifier = MLP(
            in_channels=latent_dim,
            hidden_layers=class_layers,
            activation=activation,
            norm_layer=norm_layer,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(encoder_layers[-1], latent_dim), self.classifier, nn.Linear(class_layers[-1], class_out_channels)
        )

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)

        return log_pxz.sum(-1)

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def sample_z(self, mu, logvar):
        # Reparametrization Trick to allow gradients to backpropagate from the
        # stochastic part of the model
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        latent = self.encoder(x)
        mu = self.hidden2mu(latent)
        logvar = self.hidden2log_var(latent)

        std = torch.exp(logvar / 2)

        #Sample from distribution
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        #Push sample through decoder
        decoder_output = self.decoder(z)

        class_output = self.classifier(latent)

        return mu, std, z, decoder_output, class_output

    def training_step(self, batch, batch_idx):
        
        x, y = batch
        x = x.view(x.size(0), -1)
        mu, std, z, decoder_o, class_o = self(x)

        loss = 0.0

        class_weight = self.hparams.loss["class_weight"]
        reconst_weight = self.hparams.loss["reconst_weight"]
        l1_weight = self.hparams.loss["l1_weight"]


        """if class_weight > 0:
            loss += class_weight*F.cross_entropy(class_o, y, weight=Tensor(self.w).to(self.device))
            #loss += class_weight*F.cross_entropy(class_o, y) # if not weighting for imbalanced samples.
        if reconst_weight > 0:
            loss += reconst_weight*F.mse_loss(decoder_o, x)

        if l1_weight > 0:
            loss += l1_weight*torch.abs(latent).mean()"""

        # Classification loss with weights parameter
        class_loss = class_weight * F.cross_entropy(class_o, y, weight=Tensor(self.w).to(self.device))
        # Reconstruction loss
        reconst_loss = reconst_weight * self.gaussian_likelihood(decoder_o, self.log_scale, x)

        kl = self.kl_divergence(z, mu, std)

        elbo = kl * kl_weight - reconst_loss 

        elbo = elbo.mean()

        loss = elbo + class_loss

        self.train_class = class_loss
        self.train_loss(loss, x.size(0))
        self.train_acc(class_o, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)

        mu, std, z, decoder_o, class_o = self(x)

        loss = 0.0

        class_weight = self.hparams.loss["class_weight"]
        reconst_weight = self.hparams.loss["reconst_weight"]
        l1_weight = self.hparams.loss["l1_weight"]

        """if class_weight > 0:
            loss += class_weight*F.cross_entropy(class_o, y)
            #loss += class_weight*F.cross_entropy(class_o, y, weight=Tensor(self.w))

        if reconst_weight > 0:
            loss += reconst_weight*F.mse_loss(decoder_o, x)

        #if l1_weight > 0:
         #   loss += l1_weight*torch.abs(latent).mean()"""

        # Reconstruction loss
        reconst_loss = self.gaussian_likelihood(decoder_o, self.log_scale, x)

        # expectation under z of the kl divergence between q(z|x) and
        #a standard normal distribution of the same shape
        #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl = self.kl_divergence(z, mu, std)

        elbo = kl * kl_weight - reconst_loss 

        elbo = elbo.mean()

        loss = elbo + F.cross_entropy(class_o, y)

        self.val_class = F.cross_entropy(class_o, y)
        self.elbo = elbo
        self.val_loss(loss, x.size(0))
        self.val_acc(class_o, y)

        return decoder_o, loss


    def on_train_epoch_end(self):
        loss = self.train_loss.compute()
        train_acc = self.train_acc.compute()

        self.log("hp/train_loss", loss, prog_bar=True)
        self.log("hp/train_acc", train_acc, prog_bar=True)
        self.log("hp/train_elbo", self.elbo)
        self.log("hp/train_class", self.train_class)

        self.train_loss.reset()
        self.train_acc.reset()

    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()
        acc = self.val_acc.compute()

        self.log("hp/val_loss", loss, prog_bar=True)
        self.log("hp/val_acc", acc, prog_bar=True)

        self.val_loss.reset()
        self.val_acc.reset()

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        mu, logvar, latent, decoder_o, class_o = self(x)

        #mu, std, z, decoder_output, class_output

        return latent, decoder_o, class_o


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

    def interpolate(self, x1, x2):
        assert x1.shape == x2.shape, "Inputs must be of the same shape"
        if x1.dim() == 3:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 3:
            x2 = x2.unsqueeze(0)
        if self.training:
            raise Exception(
                "This function should not be called when model is still "
                "in training mode. Use model.eval() before calling the "
                "function")
        mu1, lv1 = self.encode(x1)
        mu2, lv2 = self.encode(x2)
        z1 = self.reparametrize(mu1, lv1)
        z2 = self.reparametrize(mu2, lv2)
        weights = torch.arange(0.1, 0.9, 0.1)
        intermediate = [self.decode(z1)]
        for wt in weights:
            inter = (1.-wt)*z1 + wt*z2
            intermediate.append(self.decode(inter))
        intermediate.append(self.decode(z2))
        out = torch.stack(intermediate, dim=0).squeeze(1)
        return out, (mu1, lv1), (mu2, lv2)

    @staticmethod
    def custom_transform(normalization):
        return None, None




class LitNet(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int,
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
        onecycle_lr_scheduler: dict,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.train_loss = torchmetrics.MeanMetric()
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=class_out_channels, average="micro"
        )
        self.val_loss = torchmetrics.MeanMetric()
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=class_out_channels, average="micro"
        )
        
        self.encoder = MLP(
            in_channels=in_channels,
            hidden_layers=encoder_layers,
            activation=activation,
            norm_layer=norm_layer,
            dropout=dropout,
        )

        self.hidden2mu = nn.Linear(encoder_layers[-1], latent_dim)
        self.hidden2log_var = nn.Linear(encoder_layers[-1], latent_dim)

        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        self.decoder = MLP(
            in_channels=latent_dim,
            hidden_layers=decoder_layers,
            activation=activation,
            norm_layer=norm_layer,
            dropout=dropout,
        )

        self.decoder = nn.Sequential(
            self.decoder, nn.Linear(decoder_layers[-1], in_channels)
        )

        self.classifier = MLP(
            in_channels=latent_dim,
            hidden_layers=class_layers,
            activation=activation,
            norm_layer=norm_layer,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(encoder_layers[-1], latent_dim), self.classifier, nn.Linear(class_layers[-1], class_out_channels)
        )

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)

        return log_pxz.sum(-1)

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def sample_z(self, mu, logvar):
        # Reparametrization Trick to allow gradients to backpropagate from the
        # stochastic part of the model
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        

        latent = self.encoder(x)
        mu = self.hidden2mu(latent)
        logvar = self.hidden2log_var(latent)

        std = torch.exp(logvar / 2)

        #Sample from distribution
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        #Push sample through decoder
        decoder_output = self.decoder(z)

        class_output = self.classifier(latent)

        return mu, std, z, decoder_output, class_output

    def training_step(self, batch, batch_idx):
        
        x, y = batch
        x = x.view(x.size(0), -1)

        mu, std, z, decoder_o, class_o = self(x)

        loss = 0.0

        class_weight = self.hparams.loss["class_weight"]
        reconst_weight = self.hparams.loss["reconst_weight"]
        l1_weight = self.hparams.loss["l1_weight"]


        #if class_weight > 0:
        #    loss += class_weight*F.cross_entropy(class_o, y)
            #loss += class_weight*F.cross_entropy(class_o, y) # if not weighting for imbalanced samples.
        #if reconst_weight > 0:
         #   loss += reconst_weight*F.mse_loss(decoder_o, x)

        #if l1_weight > 0:
         #   loss += l1_weight*torch.abs(latent).mean()

        # Reconstruction loss
        reconst_loss = self.gaussian_likelihood(decoder_o, self.log_scale, x)

        # expectation under z of the kl divergence between q(z|x) and
        #a standard normal distribution of the same shape
        #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl = self.kl_divergence(z, mu, std)

        elbo = kl * 0.01 - reconst_loss 

        elbo = elbo.mean()

        loss = elbo + F.cross_entropy(class_o, y)

        self.train_loss(loss, x.size(0))
        self.train_acc(class_o, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)

        mu, std, z, decoder_o, class_o = self(x)

        loss = 0.0

        class_weight = self.hparams.loss["class_weight"]
        reconst_weight = self.hparams.loss["reconst_weight"]
        l1_weight = self.hparams.loss["l1_weight"]

        """if class_weight > 0:
            loss += class_weight*F.cross_entropy(class_o, y)
            #loss += class_weight*F.cross_entropy(class_o, y, weight=Tensor(self.w))

        if reconst_weight > 0:
            loss += reconst_weight*F.mse_loss(decoder_o, x)

        #if l1_weight > 0:
         #   loss += l1_weight*torch.abs(latent).mean()"""

        # Reconstruction loss
        reconst_loss = self.gaussian_likelihood(decoder_o, self.log_scale, x)

        # expectation under z of the kl divergence between q(z|x) and
        #a standard normal distribution of the same shape
        #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl = self.kl_divergence(z, mu, std)

        elbo = kl * 0.001 - reconst_loss 

        elbo = elbo.mean()

        loss = elbo + F.cross_entropy(class_o, y)

        self.class_loss = F.cross_entropy(class_o, y)
        self.elbo = elbo
        self.val_loss(loss, x.size(0))
        self.val_acc(class_o, y)

        return decoder_o, loss

    def on_train_epoch_end(self):
        loss = self.train_loss.compute()
        train_acc = self.train_acc.compute()

        self.log("hp/train_loss", loss, prog_bar=True)
        self.log("hp/train_acc", train_acc, prog_bar=True)
        self.log("hp/train_elbo", self.elbo)
        self.log("hp/train_class", self.class_loss)

        self.train_loss.reset()
        self.train_acc.reset()

    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()
        acc = self.val_acc.compute()

        self.log("hp/val_loss", loss, prog_bar=True)
        self.log("hp/val_acc", acc, prog_bar=True)

        self.val_loss.reset()
        self.val_acc.reset()

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        mu, logvar, latent, decoder_o, class_o = self(x)

        #mu, std, z, decoder_output, class_output

        return latent, decoder_o, class_o

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


    def interpolate(self, x1, x2):
        assert x1.shape == x2.shape, "Inputs must be of the same shape"
        if x1.dim() == 3:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 3:
            x2 = x2.unsqueeze(0)
        if self.training:
            raise Exception(
                "This function should not be called when model is still "
                "in training mode. Use model.eval() before calling the "
                "function")
        mu1, lv1 = self.encode(x1)
        mu2, lv2 = self.encode(x2)
        z1 = self.reparametrize(mu1, lv1)
        z2 = self.reparametrize(mu2, lv2)
        weights = torch.arange(0.1, 0.9, 0.1)
        intermediate = [self.decode(z1)]
        for wt in weights:
            inter = (1.-wt)*z1 + wt*z2
            intermediate.append(self.decode(inter))
        intermediate.append(self.decode(z2))
        out = torch.stack(intermediate, dim=0).squeeze(1)
        return out, (mu1, lv1), (mu2, lv2)

    @staticmethod
    def custom_transform(normalization):
        return None, None