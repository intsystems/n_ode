""" Example script for test trajectories classification.
    Editable.
"""
import re
import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import numpy as np

import torch
import torch.nn.functional as F

import wandb
from wandb.util import generate_id

import lightning as L
from lightning.pytorch.loggers import WandbLogger

from node.field_model import VectorFieldMLP
from node.data_modules import ActivityDataModule
from components.field_module import LitNodeHype
from components.classify import build_lh_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subj_id", type=int)
    parser.add_argument("datasets_dir", type=Path)
    parser.add_argument("checkpoints_dir", type=Path)
    parser.add_argument("wandb_config", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    # load config file
    wandb_config: DictConfig = OmegaConf.load(args.wandb_config)
    subj_id = args.subj_id

    run = wandb.init(
        name="classify-" + generate_id(),
        tags=[f"subj{subj_id}"],
        **dict(wandb_config)
    )

    test_datasets = {}
    for act_data_dir in args.datasets_dir.glob("*"):
        act = act_data_dir.name
        test_datasets[act] = torch.load(act_data_dir / "test.pkl", weights_only=False)
        run.use_artifact(f"subj_{subj_id}_{act}_test:latest")

    models = {}
    for act in test_datasets.keys():
        train_config = OmegaConf.load(f"config/{act}/train.yaml")
        checkpoint_artifact = run.use_artifact(f"subj_{subj_id}_{act}_checkpoint:latest")
        # example model loading
        lit_node = LitNodeHype.load_from_checkpoint(
            checkpoint_path=args.checkpoints_dir / f"{act}_checkpoint.ckpt",
            vf=VectorFieldMLP(**dict(train_config.vf)),
            loss_funcs={
                "MSE": F.mse_loss,
                "MAE": F.l1_loss
            }
        )
        models[act] = lit_node

    # compute lh table
    lh_df = build_lh_table(test_datasets, models)
    lh_df["subj"] = subj_id
    lh_df.to_csv(args.output_dir / f"subj_{subj_id}_classify_table.csv", index=False)
    # save lh table
    run.log_artifact(
        args.output_dir / f"subj_{subj_id}_classify_table.csv",
        f"subj_{subj_id}_classify_table"
    )
