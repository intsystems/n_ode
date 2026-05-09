import argparse
import os
from pathlib import Path
from omegaconf import OmegaConf
from itertools import chain
from toolz import pipe
from toolz.curried import map as map_c

import pandas as pd

import torch
from torch.utils.data import DataLoader, ConcatDataset
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
# from torchmetrics.regression import ...

from experiment.motion_sense.utils.dataset import TrajectoryDataset
from experiment.motion_sense.utils.field import FieldLitModule

import mlflow

BATCH_SIZE = 32
NOISE_SIGMA = 1e-2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("act", type=str)
    parser.add_argument("subj", type=int)
    args = parser.parse_args()
    config = OmegaConf.load("experiment/motion_sense/config.yaml")

    train_dataset = [
        TrajectoryDataset(
            config.data_dir, config.data_types, args.act, train_act_code, args.subj
        )
        for train_act_code in config.activity_codes[args.act][:-1]
    ]
    train_dataset = ConcatDataset(train_dataset)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    test_dataset = TrajectoryDataset(
        config.data_dir, config.data_types, args.act,
        config.activity_codes[args.act][-1], args.subj
    )
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

    state_noise_sigma = pd.read_csv(
        os.path.join(config.results_dir, str(args.subj), args.act, "traj_std.csv"),
        index_col=0
    ).to_numpy().flatten()
    state_dim = test_dataset[0].shape[0] // 2
    state_names = pipe(
        config.data_types,
        map_c(lambda name: [name + '.' + suffix for suffix in ['x', 'y', 'z']]),
        chain.from_iterable,
        list
    )
    field_module = FieldLitModule(
        state_dim, config.dt, state_noise_sigma, NOISE_SIGMA, state_names
    )

    logger = MLFlowLogger(
        experiment_name="motion_sense", tracking_uri=config.tracking_uri,
        run_name="learn_field",
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
        callbacks=[checkpointing],
        logger=logger,
        max_epochs=3,
        log_every_n_steps=20
    )
    trainer.fit(
        field_module, train_loader, test_loader
    )

