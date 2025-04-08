""" Example script for test trajectories classification.
    Editable.
"""
import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import numpy as np

import torch

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from node.field_model import VectorFieldMLP
from node.data_modules import ActivityDataModule
from components.field_module import LitNodeHype
from components.classify import build_lh_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="classify_config.yaml",
                        required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    # load config file
    config: DictConfig = OmegaConf.load(args.config_path)
    # transform some fields to correct types
    config.data.data_path = Path(config.data.data_path)
    config.save_dir = Path(config.save_dir)

    models = {}
    for act in config.activities:
        model_path = WandbLogger.download_artifact(
                f"kirill-semkin32-forecsys/node/{act}_model:latest",
                artifact_type="model"
        ) + "/model.ckpt"
        # example model loading
        lit_node = LitNodeHype.load_from_checkpoint(
            model_path,
            torch.device("cpu"),
            vf=VectorFieldMLP(**dict(config.vf)),
        )
        models[act] = lit_node

    # create lightning data modules
    # trajectories will be stored in a temporary files as they are not train artifacts
    datamodules = {}
    for act in config.activities:
        activity_data = ActivityDataModule(
            act,
            save_dir=config.save_dir / act,
            **dict(config.data)
        )
        activity_data.prepare_data()
        activity_data.setup(stage="val")
        datamodules[act] = activity_data

    # compute lh table
    lh_df = build_lh_table(datamodules, models)
    # save lh table
    ...

    # make analytics
    lh_df["best_model"] = lh_df.drop(columns=["test_act", "traj_num"]).idxmin(axis=1)
    lh_df["is_correct_cls"] = (lh_df["best_model"] == lh_df["test_act"]).astype(np.int32)
    accuracy_per_act = lh_df.groupby("test_act")["is_correct_cls"].mean()
    print("Accuracy per activity class", accuracy_per_act)
