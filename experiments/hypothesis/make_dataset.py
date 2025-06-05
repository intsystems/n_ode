""" Example script to launch dataset building.
    Editable.
"""
import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from node.field_model import VectorFieldMLP
from node.data_modules import ActivityDataModule
from components.field_module import LitNodeHype


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_config")
    parser.add_argument("data_path", type=Path)
    parser.add_argument("save_dir", type=Path)
    args = parser.parse_args()

    print(args)

    # load config file
    data_config: DictConfig = OmegaConf.load(args.data_config)

    # build trajectories
    activity_data = ActivityDataModule(
        data_path=args.data_path,
        save_dir=args.save_dir,
        **dict(data_config.data)
    )
    activity_data.prepare_data()
