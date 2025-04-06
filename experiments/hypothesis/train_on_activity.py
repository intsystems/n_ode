""" Script to launch model training
    Editable
"""
import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from node.field_model import VectorFieldMLP
from field_module import LitNodeHype
from node.data_modules import ActivityDataModule


def train_on_activity(
    config: DictConfig
):
    vf = VectorFieldMLP(**dict(config.vf))
    lit_node = LitNodeHype(vf, dict(config.optim), dict(config.odeint))

    # create lightning data module
    # trajectories will be stored in a temporary files as they are not train artifacts
    activity_data = ActivityDataModule(
        **dict(config.data)
    )

    logger = WandbLogger(
        project="node",
        group="hypothesis",
        tags=["train", "unnormalized", config.data.act, "mlp_tanh"],
        config=dict(config),
        log_model="all",
        # mode="disabled" # debug
    )

    checkpoint_callback = ModelCheckpoint(monitor="Val/MSE", mode="min")
    trainer = L.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        **dict(config.trainer)
    )

    trainer.fit(lit_node, datamodule=activity_data)


if __name__ == "__main__":
    # possibility to run as separate script
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="config.yaml",
                        required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    # load config file
    config: DictConfig = OmegaConf.load(args.config_path)
    # transform some fields to correct types
    config.data.data_path = Path(config.data.data_path)
    config.data.save_dir = Path(config.data.save_dir)

    train_on_activity(config)
