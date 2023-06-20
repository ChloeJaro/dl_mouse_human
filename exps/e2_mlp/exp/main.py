import os

import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf
import hydra

from model import LitNet
from dataset import MouseHumanDataModule, encode
from utils import save_config

THIS_PATH = os.path.realpath(os.path.dirname(__file__))
RESULTS_PATH = os.path.join(THIS_PATH, "../results.ign/")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    tag = cfg["tag"]

    fast_dev_run = cfg["trainer"]["fast_dev_run"]

    exp_root = os.path.join(RESULTS_PATH, tag)

    if not fast_dev_run:
        os.makedirs(exp_root)

        save_config(cfg, path=os.path.join(exp_root, "config.yaml"))

    cfg = OmegaConf.to_container(cfg)

    seed_everything(cfg["seed"], workers=True)

    model = LitNet(**cfg["model"])
    data = MouseHumanDataModule(**cfg["data"])

    logger = TensorBoardLogger(save_dir=exp_root)
    checkpoint_callback = ModelCheckpoint(dirpath=exp_root, save_last=True)

    trainer = pl.Trainer(
        default_root_dir=exp_root,
        callbacks=[checkpoint_callback],
        logger=logger,
        **cfg["trainer"]
    )

    trainer.fit(model=model, datamodule=data)

    if fast_dev_run:
        return

    ckpt_path = os.path.join(exp_root, "last.ckpt")

    encode_path = os.path.join(exp_root, "encoding")
    os.makedirs(encode_path)

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
        encode(trainer=trainer, ckpt_path=ckpt_path,
               **enc_recipe, **cfg['encode'])


if __name__ == "__main__":
    main()
