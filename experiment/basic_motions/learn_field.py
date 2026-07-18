import argparse
import os
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from experiment.basic_motions.utils.dataset import TrajectoryDataset
from experiment.basic_motions.utils.field import FieldLitModule

BATCH_SIZE = 256
NUM_WORKERS = 2
WINDOW_SIZE = 20


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label", type=str)
    args = parser.parse_args()
    config = OmegaConf.load("experiment/basic_motions/config.yaml")

    train_dataset = TrajectoryDataset(
        config.train_path, args.label,
        config.gyro_channels, config.accel_channels,
        dt=config.dt, window_size=WINDOW_SIZE
    )
    test_dataset = TrajectoryDataset(
        config.test_path, args.label,
        config.gyro_channels, config.accel_channels,
        dt=config.dt, window_size=WINDOW_SIZE
    )

    traj_mean = torch.concat(train_dataset.trajs).mean(dim=0).to(dtype=torch.float32)
    traj_std = torch.concat(train_dataset.trajs).std(dim=0).to(dtype=torch.float32)
    continuous_controls_mean = torch.concat(train_dataset.controls).mean(dim=0).to(dtype=torch.float32)
    continuous_controls_std = torch.concat(train_dataset.controls).std(dim=0).to(dtype=torch.float32)
    assert continuous_controls_std.shape == (3,)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    field_module = FieldLitModule(
        train_dataset.d, train_dataset.num_cont_controls, config.dt,
        traj_mean, traj_std,
        continuous_controls_mean, continuous_controls_std
    )

    logger = MLFlowLogger(
        experiment_name="basic_motions", tracking_uri=config.tracking_uri,
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
        min_delta=1e-3, patience=10
    )
    trainer = Trainer(
        accelerator="auto",
        callbacks=[checkpointing, early_stop],
        logger=logger,
        max_epochs=100,
        log_every_n_steps=3
    )
    trainer.fit(
        field_module, train_loader, test_loader
    )
