import argparse
import os
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from experiment.basic_motions_sde_no_control.utils.dataset import TrajectoryDataset
from experiment.basic_motions_sde_no_control.utils.field import FieldLitModule

BATCH_SIZE = 1600
NUM_WORKERS = 1
WINDOW_SIZE = 20


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label", type=str)
    args = parser.parse_args()
    config = OmegaConf.load("experiment/basic_motions_sde_no_control/config.yaml")

    train_dataset = TrajectoryDataset(
        config.train_path, args.label,
        config.gyro_channels,
        dt=config.dt, window_size=WINDOW_SIZE
    )
    test_dataset = TrajectoryDataset(
        config.test_path, args.label,
        config.gyro_channels,
        dt=config.dt, window_size=WINDOW_SIZE
    )

    traj_mean = torch.concat(train_dataset.trajs).mean(dim=0).to(dtype=torch.float32)
    traj_std = torch.concat(train_dataset.trajs).std(dim=0).to(dtype=torch.float32)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    field_module = FieldLitModule(
        train_dataset.d, config.dt,
        traj_mean, traj_std
    )

    logger = MLFlowLogger(
        experiment_name="basic_motions_sde_no_control", tracking_uri=config.tracking_uri,
        run_name="learn_field",
        log_model="all",
        tags={"label": args.label}
    )
    checkpointing = ModelCheckpoint(
        os.path.join(config.results_dir, args.label),
        filename="best", monitor="Val/loss", mode="min",
        enable_version_counter=False, save_last=True
    )
    early_stop = EarlyStopping(
        monitor="Val/loss",
        min_delta=1e-3, patience=20
    )
    trainer = Trainer(
        accelerator="auto",
        callbacks=[checkpointing, early_stop],
        logger=logger,
        max_epochs=400,
        log_every_n_steps=1
    )
    trainer.fit(
        field_module, train_loader, test_loader,
        # ckpt_path="results_basic_motions_sde_no_control/Standing/last.ckpt", weights_only=False
    )
