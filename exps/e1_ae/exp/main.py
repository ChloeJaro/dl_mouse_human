import os
import torch
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf
import hydra

from torch import Tensor
from torch.utils.data import TensorDataset, Subset
import numpy as np

from model import LitNet, LitNet_w
from dataset import MouseHumanDataModule, MouseHumanDataModuleCV, GetWeights, encode, autoencode, classify
from utils import save_config

THIS_PATH = os.path.realpath(os.path.dirname(__file__))
RESULTS_PATH = os.path.join(THIS_PATH, "../results.ign/")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    tag = cfg["tag"]

    fast_dev_run = cfg["trainer"]["fast_dev_run"]

    balance_weighting = cfg["weights"]

    cross_val = cfg["cross_val"]

    exp_root = os.path.join(RESULTS_PATH, tag)

    if not fast_dev_run:
        if not os.path.exists(exp_root):
            os.makedirs(exp_root)

        save_config(cfg, path=os.path.join(exp_root, "config.yaml"))
    
    encode_path = os.path.join(exp_root, "encoding")

    cfg = OmegaConf.to_container(cfg)
    early_stopping = EarlyStopping(monitor='hp/train_loss', patience=3, verbose=True)


    if cross_val:
        data = MouseHumanDataModuleCV(seed=cfg["seed"], **cfg["data"])

    else:
        data = MouseHumanDataModule(seed=cfg["seed"], **cfg["data"])

    logger = TensorBoardLogger(save_dir=exp_root)
    checkpoint_callback = ModelCheckpoint(dirpath=exp_root, save_last=True)

    seed_everything(cfg["seed"], workers=True)

    if balance_weighting:
    
        weights = GetWeights(ref_data_path=cfg["data"]["coronal_maskcoronal_path"], labelcol=cfg["data"]["mouse_labelcol"])
        weights_ = weights.balance_weights
        model = LitNet_w(weights_, **cfg["model"])
    
    else:
        model = LitNet(**cfg["model"])

        
    trainer = pl.Trainer(
        default_root_dir=exp_root,
        callbacks=[checkpoint_callback],
        logger=logger,
        **cfg["trainer"]
    )
    
    if not os.path.exists(encode_path):
        os.makedirs(encode_path)

    trainer.fit(model=model, datamodule=data)


    if fast_dev_run:
        return

    ckpt_path = os.path.join(exp_root, "last.ckpt")


    model_best = LitNet_w.load_from_checkpoint(ckpt_path)

    encode_recipes = [
        {
            'data_path': cfg["data"]["mouse_voxel_data_path"],
            'intersct_data_path': cfg["data"]["human_voxel_data_path"],
            'labelcol': cfg["data"]["mouse_labelcol"],
            'output_file_path': os.path.join(encode_path, "mouse_voxel_encoding.csv"),
        },
        {
            'data_path': cfg["data"]["human_voxel_data_path"],
            'intersct_data_path': cfg["data"]["mouse_voxel_data_path"],
            'labelcol': cfg["data"]["human_labelcol"],
            'output_file_path': os.path.join(encode_path, "human_voxel_encoding.csv"),
        },
        {
            'data_path': cfg["data"]["mouse_region_data_path"],
            'intersct_data_path': cfg["data"]["human_voxel_data_path"],
            'labelcol': 'Region',
            'output_file_path': os.path.join(encode_path, "mouse_region_encoding.csv"),
        },
        {
            'data_path': cfg["data"]["human_region_data_path"],
            'intersct_data_path': cfg["data"]["mouse_voxel_data_path"],
            'labelcol': 'Region',
            'output_file_path': os.path.join(encode_path, "human_region_encoding.csv"),
        },
    ]

    
    for enc_recipe in encode_recipes:
        encode(
            trainer=trainer, 
            model=model_best,
            ckpt_path=ckpt_path, 
            ref_data_path=cfg["data"]["coronal_maskcoronal_path"],
            cor_data_path=cfg["data"]["coronal_masksagittal_path"],
            sag_data_path=cfg["data"]["sagittal_masksagittal_path"],
            seed=cfg["seed"],
            **enc_recipe, **cfg['encode'])


    #if os.path.exists(encode_path):

    autoencode_path = os.path.join(exp_root, "autoencoding")
    if not os.path.exists(autoencode_path):
        os.makedirs(autoencode_path)
    ckpt_path = os.path.join(exp_root, "last.ckpt")
    #ckpt_path = checkpoint_callback.best_model_path # chose best model encountered during training
    #model_best = model.load_from_checkpoint(ckpt_path, map_location=torch.device('cpu')) # on local machine
    #model_best = model.load_from_checkpoint(ckpt_path)
    autoencode_recipes = [
    {
        'data_path': cfg["data"]["mouse_voxel_data_path"],
        'intersct_data_path': cfg["data"]["human_voxel_data_path"],
        'labelcol': cfg["data"]["mouse_labelcol"],
        'output_file_path': os.path.join(autoencode_path, "mouse_voxel_autoencoding.csv"),
    },
    {
        'data_path': cfg["data"]["human_voxel_data_path"],
        'intersct_data_path': cfg["data"]["mouse_voxel_data_path"],
        'labelcol': cfg["data"]["human_labelcol"],
        'output_file_path': os.path.join(autoencode_path, "human_voxel_autoencoding.csv"),
    },
    {
        'data_path': cfg["data"]["mouse_region_data_path"],
        'intersct_data_path': cfg["data"]["human_voxel_data_path"],
        'labelcol': 'Region',
        'output_file_path': os.path.join(autoencode_path, "mouse_region_autoencoding.csv"),
    },
    {
        'data_path': cfg["data"]["human_region_data_path"],
        'intersct_data_path': cfg["data"]["mouse_voxel_data_path"],
        'labelcol': 'Region',
        'output_file_path': os.path.join(autoencode_path, "human_region_autoencoding.csv"),
    },
    ]

   # trainer.fit(model_best,data)
    for autoenc_recipe in autoencode_recipes:
        autoencode(
            trainer=trainer, 
            model=model_best,
            ckpt_path=ckpt_path,
            ref_data_path=cfg["data"]["coronal_maskcoronal_path"],
            cor_data_path=cfg["data"]["coronal_masksagittal_path"],
            sag_data_path=cfg["data"]["sagittal_masksagittal_path"],
            seed=cfg["seed"],
            **autoenc_recipe, **cfg['encode'])

    classify_path = os.path.join(exp_root, "classifying")
    if not os.path.exists(classify_path):
        os.makedirs(classify_path)
    
    classify_recipes = [
        {
            'data_path': cfg["data"]["mouse_voxel_data_path"], 
            'intersct_data_path': cfg["data"]["human_voxel_data_path"],
            'labelcol': "Region67",
            'output_file_path': os.path.join(classify_path, "mouse_voxel_classification.csv"),
            'labels_file_path': os.path.join(classify_path, "mouse_voxel_in_labels.csv"),
        },
        {
            'data_path': cfg["data"]["human_voxel_data_path"],
            'intersct_data_path': cfg["data"]["mouse_voxel_data_path"],
            'labelcol': "Region88",
            'output_file_path': os.path.join(classify_path, "human_voxel_classification.csv"),
            'labels_file_path': os.path.join(classify_path, "mouse_voxel_in_labels.csv"),
        },
    ]

   # trainer.fit(model_best,data)
    for classif_recipe in classify_recipes:
        classify(
            trainer=trainer, 
            model=model_best,
            ckpt_path=ckpt_path,
            ref_data_path=cfg["data"]["coronal_maskcoronal_path"],
            cor_data_path=cfg["data"]["coronal_masksagittal_path"],
            sag_data_path=cfg["data"]["sagittal_masksagittal_path"],
            seed=cfg["seed"],
            **classif_recipe, **cfg['encode'])

if __name__ == "__main__":
    main()
