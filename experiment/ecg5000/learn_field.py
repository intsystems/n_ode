import argparse
import os
from omegaconf import OmegaConf


import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from experiment.ecg5000.utils.dataset import TakensSlicedTrajectoryDataset
from experiment.ecg5000.utils.field import FieldLitModule

BATCH_SIZE = 256
NUM_WORKERS = 1


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
    traj_mean = torch.zeros((config.delay_dim,), dtype=torch.float32)
    traj_std = torch.ones((config.delay_dim,), dtype=torch.float32)

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
    early_stop = EarlyStopping(
        monitor="Val/loss",
        patience=7
    )
    trainer = Trainer(
        accelerator="gpu",
        callbacks=[checkpointing, early_stop],
        logger=logger,
        max_epochs=100,
        log_every_n_steps=10
    )
    trainer.fit(
        field_module, train_loader, test_loader
    )
