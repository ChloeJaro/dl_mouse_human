import os

import torch
import torch.optim as optim

#import lightning.pytorch as pl
import pytorch_lightning as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf
import hydra

from functools import partial

from ray import air, tune
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining 
#from ray.tune.integration.pytorch_lightning import TuneReportCallback 
from ray.train.lightning import LightningTrainer, LightningConfigBuilder


from model import LitNet
from dataset import MouseHumanDataModule, encode
from utils import save_config

THIS_PATH = os.path.realpath(os.path.dirname(__file__))
RESULTS_PATH = os.path.join(THIS_PATH, "../results.ign/")





@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    num_epochs = 5
    num_samples = 10
    accelerator="gpu"
    
    tag = cfg["tag"]

    fast_dev_run = cfg["trainer"]["fast_dev_run"]

    exp_root = os.path.join(RESULTS_PATH, tag)


    if not fast_dev_run:
        os.makedirs(exp_root)

        save_config(cfg, path=os.path.join(exp_root, "config.yaml"))

    cfg = OmegaConf.to_container(cfg)
    default_config = cfg
    seed_everything(cfg["seed"], workers=True)

    model = LitNet(**cfg["model"])
    data = MouseHumanDataModule(**cfg["data"])

    logger = TensorBoardLogger(save_dir=exp_root)

    static_lightning_config = (
        LightningConfigBuilder()
        .module(cls=LitNet)
        .trainer(max_epochs=num_epochs, accelerator=accelerator, logger=logger)
        .fit_params(datamodule=data)
        .checkpointing(monitor="val_acc", save_top_k=2, mode="max")
        .build()
    )

    searcheable_lightning_config = (
        LightningConfigBuilder()
        .module(config = {
            "model.encoder_layers": tune.grid_search([[200, 200], [500, 500], [1000, 1000]]),
            "model.decoder_layers": tune.grid_search([[200, 200], [300, 300], [1000, 1000]]),
            "model.class_layers": tune.grid_search([[200, 200], [300, 300], [1000, 1000]]),
            "model.norm_layer.name": tune.choice(["batch", "instance", "none"]),
            "model.dropout": tune.grid_search([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            "model.loss.l1_weight": tune.grid_search([0.01, 0.1, 0.5, 1, 10]),
            "model.loss.reconst_weight": tune.grid_search([0.01, 0.1, 0.5, 1, 10]),

        })    
    )
    
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val_acc",
            checkpoint_score_order="max",
        ),
    )

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    scaling_config = ScalingConfig(
        num_workers=3, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
    )
    #trainer.fit(model=model, datamodule=data)
    lightning_trainer = LightningTrainer(
        lightning_config=static_lightning_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    def tune_asha(num_samples=10):
        scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

        tuner = tune.Tuner(
            lightning_trainer,
            param_space={"lightning_config": searcheable_lightning_config},
            tune_config=tune.TuneConfig(
                metric="val_acc",
                mode="max",
                num_samples=num_samples,
                scheduler=scheduler,
            ),
            run_config=air.RunConfig(
                name="tune_mnist_asha",
            ),
        )
        results = tuner.fit()
        best_result = results.get_best_result(metric="val_acc", mode="max")
        best_result

    tune_asha(num_samples=num_samples)




if __name__ == "__main__":
    main()
