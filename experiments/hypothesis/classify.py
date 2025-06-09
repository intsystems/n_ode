""" Example script for test trajectories classification.
    Editable.
"""
import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import numpy as np

import torch
import wandb

import lightning as L
from lightning.pytorch.loggers import WandbLogger

from node.field_model import VectorFieldMLP
from node.data_modules import ActivityDataModule
from components.field_module import LitNodeHype
from components.classify import build_lh_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("classify_config", type=Path)
    parser.add_argument("wandb_config", type=Path)
    args = parser.parse_args()

    # load config file
    classify_config: DictConfig = OmegaConf.load(args.classify_config)
    wandb_config: DictConfig = OmegaConf.load(args.wandb_config)

    run = wandb.init(
        **dict(wandb_config),
        tags=["classify"],
        #mode="disabled"     # DEBUG
    )

    models = {}
    for act, model_name in classify_config.items():
        train_config = OmegaConf.load(f"config/{act}/train.yaml")
        checkpoint_artifact = run.use_artifact(model_name)
        model_path = checkpoint_artifact.download()
        # example model loading
        lit_node = LitNodeHype.load_from_checkpoint(
            Path(model_path) / "model.ckpt",
            torch.device("cpu"),
            vf=VectorFieldMLP(**dict(train_config.vf)),
        )
        models[act] = lit_node

    # create lightning data modules
    # trajectories will be stored in a temporary files as they are not train artifacts
    datamodules = {}
    for act in classify_config:
        activity_data = ActivityDataModule(
            save_dir=Path("act_data") / act
        )
        activity_data.setup(stage="val")
        datamodules[act] = activity_data

    # compute lh table
    lh_df = build_lh_table(datamodules, models)
    # save lh table
    run.log({"classify_table": wandb.Table(dataframe=lh_df)})

    # make small analytics
    lh_df["best_model"] = lh_df.drop(columns=["test_act", "traj_num"]).idxmin(axis=1)
    lh_df["is_correct_cls"] = (lh_df["best_model"] == lh_df["test_act"]).astype(np.int32)
    accuracy_per_act = lh_df.groupby("test_act")["is_correct_cls"].mean()
    print("Accuracy per activity class", accuracy_per_act)
