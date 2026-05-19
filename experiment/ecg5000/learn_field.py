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

from experiment.ecg5000.utils.dataset import TakensSlicedTrajectoryDataset, TakensTrajectoryDataset
from experiment.ecg5000.utils.field import FieldLitModule

BATCH_SIZE = 16
NUM_WORKERS = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label", type=int)
    args = parser.parse_args()
    config = OmegaConf.load("experiment/ecg5000/config.yaml")

    train_dataset = TakensSlicedTrajectoryDataset(
        os.path.join(config.data_dir, "ECG5000_TRAIN.txt"),
        config.delay_dim, args.label, config.window_size
    )
    test_dataset = TakensSlicedTrajectoryDataset(
        os.path.join(config.data_dir, "ECG5000_TEST.txt"),
        config.delay_dim, args.label, config.window_size, max_series=100
    )
    # normalization
    unslided_train_dataset = TakensTrajectoryDataset(
        os.path.join(config.data_dir, "ECG5000_TRAIN.txt"),
        config.delay_dim, args.label
    )
    all_train_traj = torch.concat(
        [
            unslided_train_dataset[i]
            for i in range(len(unslided_train_dataset))
        ]
    )
    traj_mean = all_train_traj.mean(dim=0)
    traj_std = all_train_traj.std(dim=0)
    del unslided_train_dataset
    del all_train_traj

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

    state_dim = test_dataset.delay_dim
    field_module = FieldLitModule(
        state_dim, config.dt,
        traj_mean=traj_mean, traj_std=traj_std
    )

    logger = MLFlowLogger(
        experiment_name="ecg5000", tracking_uri=config.tracking_uri,
        run_name="learn_ode",
        log_model=True,
        tags={"label": args.label}
    )
    checkpointing = ModelCheckpoint(
        os.path.join(config.results_dir, str(args.label)),
        filename="best", monitor="Val/loss", mode="min",
        enable_version_counter=False
    )
    trainer = Trainer(
        accelerator="cpu",
        # devices=4,
        callbacks=[checkpointing],
        logger=logger,
        max_epochs=1,
        log_every_n_steps=10
    )
    trainer.fit(
        field_module, train_loader, test_loader
    )
