import argparse
import os
from omegaconf import OmegaConf

import numpy as np

import torch
from torchdiffeq import odeint

from experiment.ecg5000.utils.dataset import TakensTrajectoryDataset
from experiment.ecg5000.utils.field import FieldLitModule



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label", type=int)
    args = parser.parse_args()
    config = OmegaConf.load("experiment/ecg5000/config.yaml")

    train_dataset = TakensTrajectoryDataset(
        os.path.join(config.data_dir, "ECG5000_TRAIN.txt"),
        config.delay_dim, args.label
    )

    field_adapter = FieldLitModule.load_from_checkpoint(
        os.path.join(config.results_dir, args.label, "best.ckpt"),
        weights_only=False,
        traj_mean=torch.zeros((config.delay_dim, ), dtype=torch.float32),
        traj_std=torch.zeros((config.delay_dim, ), dtype=torch.float32)
    ).to("cpu").eval()
    for param in field_adapter.parameters():
        param.requires_grad = False

    traj_deviation = []
    for i in range(len(train_dataset)):
        traj = train_dataset[i]
        t = torch.arange(traj.shape[0], dtype=torch.float32) * field_adapter.dt
        pred_traj = odeint(field_adapter.field, traj[0], t)
        traj_deviation.append(traj - pred_traj)
    traj_deviation = torch.concat(traj_deviation)
    std_est = traj_deviation.std(dim=0).numpy()
    np.save(
        os.path.join(config.results_dir, args.label, "noise_sigma.npy"),
        std_est
    )
