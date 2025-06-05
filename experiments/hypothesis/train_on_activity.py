""" Example script to launch model training.
    Editable.
"""
import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import torch

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from node.field_model import VectorFieldMLP
from node.data_modules import ActivityDataModule
from components.field_module import LitNodeHype


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_config")
    parser.add_argument("train_config")
    args = parser.parse_args()

    # load config files
    data_config: DictConfig = OmegaConf.load(args.data_config)
    train_config: DictConfig = OmegaConf.load(args.train_config)
    # transform some fields to correct types
    data_config.data.data_path = Path(data_config.data.data_path)
    data_config.data.save_dir = Path(data_config.data.save_dir)

    # set rand seed
    torch.manual_seed(train_config.seed)

    # choose vector field
    vf = VectorFieldMLP(**dict(train_config.vf))
    lit_node = LitNodeHype(vf, dict(train_config.optim), dict(train_config.odeint))

    # create lightning data module
    activity_data = ActivityDataModule(
        **dict(data_config.data)
    )

    logger = WandbLogger(
        project="node",
        group="hypothesis",
        tags=train_config.tags,
        config=dict(train_config) | dict(data_config),
        log_model="all",
        # it only attributes to wandb cloud
        checkpoint_name=f"{data_config.data.act}_checkpoint",
        # mode="disabled" # debug
    )

    checkpoint_callback = ModelCheckpoint(
        filename=f"{data_config.data.act}_checkpoint",
        #monitor="Val/MSE",
        #mode="min"
    )
    trainer = L.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        **dict(train_config.trainer)
    )

    trainer.fit(lit_node, datamodule=activity_data)
    print(checkpoint_callback.best_model_path)
