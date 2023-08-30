import os

import torch
import torch.optim as optim

import lightning.pytorch as pl
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

"""
from model import LitNet
from dataset import MouseHumanDataModule, encode
from utils import save_config """

print("test")