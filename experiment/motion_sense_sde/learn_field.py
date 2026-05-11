import argparse
import os
from pathlib import Path
from omegaconf import OmegaConf
from itertools import chain
from toolz import pipe
from toolz.curried import map as map_c

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, ConcatDataset
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
# from torchmetrics.regression import ...

from experiment.motion_sense_sde.utils.dataset import TrajectoryDataset
from experiment.motion_sense_sde.utils.field import FieldLitModule

BATCH_SIZE = 32
NUM_WORKERS = 2
WINDOW_SIZE = 16


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("act", type=str)
    parser.add_argument("subj", type=int)
    args = parser.parse_args()
    config = OmegaConf.load("experiment/motion_sense_sde/config.yaml")

    train_datasets = [
        TrajectoryDataset(
            config.data_dir, config.data_types, config.state_names,
            args.act, train_act_code, args.subj,
            window_size=WINDOW_SIZE
        )
        for train_act_code in config.activity_codes[args.act][0:1]
    ]
    test_dataset = TrajectoryDataset(
        config.data_dir, config.data_types, config.state_names,
        args.act, config.activity_codes[args.act][-1], args.subj,
        window_size=WINDOW_SIZE
    )

    # per-channel normalization: pool stats over training trials, apply to all datasets
    all_train_traj = torch.cat([ds.traj for ds in train_datasets])
    traj_mean = all_train_traj.mean(0)
    traj_std = all_train_traj.std(0)

    train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

    state_dim = test_dataset.d
    field_module = FieldLitModule(
        state_dim, config.dt, config.state_names,
        traj_mean=traj_mean, traj_std=traj_std
    )

    logger = MLFlowLogger(
        experiment_name="motion_sense_sde", tracking_uri=config.tracking_uri,
        run_name="learn_sde",
        log_model=True,
        tags={"act": args.act, "subj": str(args.subj)}
    )
    checkpointing = ModelCheckpoint(
        os.path.join(config.results_dir, str(args.subj), args.act),
        filename="best", monitor="Val/loss", mode="min",
        enable_version_counter=False
    )
    trainer = Trainer(
        accelerator="cpu",
        # devices=4,
        callbacks=[checkpointing],
        logger=logger,
        max_epochs=1,
        log_every_n_steps=20
    )
    trainer.fit(
        field_module, train_loader, test_loader
    )

