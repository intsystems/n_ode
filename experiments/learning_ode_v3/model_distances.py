""" computing KL-divergence between ML-estimators of diffrent activities
"""
import sys
from pathlib import Path
import yaml
from tqdm import tqdm
from datetime import datetime
from typing import Optional
from collections import defaultdict
import warnings

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torchdyn.core import NeuralODE
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.field_model import *
from train import get_trajectory_mask

import warnings
warnings.simplefilter("ignore", FutureWarning)


def vectorize_model_params(model: nn.Module) -> torch.Tensor:
    vec_params = []
    for param in model.parameters():
        vec_params.append(param.flatten().detach())

    return torch.concat(vec_params)


def fisher_hessian(ode_model: NeuralODE, train_loader: DataLoader) -> torch.Tensor:
    """ Computes diagonal elements of the Fisher matrix of log(p). It is factorized approximation
            of the ML inversed hessian. Returns diagonal log-std.
    """
    ode_model.zero_grad()
    ode_model.eval()

    num_batches = len(list(iter(train_loader)))

    for batch in tqdm(train_loader, desc="Accumulating grads", leave=False):
        traj: torch.Tensor = batch[0].to(ode_model.device)
        durations: torch.Tensor = batch[1].to(ode_model.device)

        traj_len = traj.shape[1]
        t_span = torch.arange(0, traj_len).to(ode_model.device)
        mask = get_trajectory_mask(durations, traj)

        t_eval, traj_predict = ode_model(traj[:, 0, :], t_span)
        # move batch axis in front
        traj_predict = traj_predict.movedim(1, 0)

        log_lh = -F.mse_loss(
            traj.flatten(end_dim=-2),
            (traj_predict * mask).flatten(end_dim=-2),
            reduction="mean"
        )
        log_lh *= (traj.numel() / durations.sum())
        log_lh /= num_batches

        # accumulate gradients
        log_lh.backward()

    # compute log std out of log(p) gradients (reverse hessian approximation)
    diag_log_std = []
    for param in ode_model.parameters():
        diag_log_std.append(
            - torch.log(param.grad.abs().flatten().detach())
        )
    diag_log_std = torch.concat(diag_log_std)

    # check for errors
    if torch.isnan(diag_log_std).sum() > 0:
        warnings.warn(
            f"{torch.isnan(diag_log_std).sum()
               } NaNs in fisher matrix, change them for zero",
            RuntimeWarning
        )
        diag_log_std = diag_log_std.nan_to_num()

    return diag_log_std


def get_normal_KL(
    mean_1: torch.Tensor,
    log_std_1: torch.Tensor,
    mean_2: Optional[torch.Tensor] = None,
    log_std_2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    :Parameters:
    mean_1: means of normal distributions (1)
    log_std_1 : standard deviations of normal distributions (1)
    mean_2: means of normal distributions (2)
    log_std_2 : standard deviations of normal distributions (2)
    :Outputs:
    kl divergence of the normal distributions (1) and normal distributions (2)
    ---
    This function should return the value of KL(p1 || p2),
    where p1 = Normal(mean_1, exp(log_std_1) ** 2), p2 = Normal(mean_2, exp(log_std_2) ** 2).
    If mean_2 and log_std_2 are None values, we will use standard normal distribution.
    Note that we consider the case of diagonal covariance matrix.
    """
    if mean_2 is None:
        mean_2 = torch.zeros_like(mean_1)
    if log_std_2 is None:
        log_std_2 = torch.zeros_like(log_std_1)
    assert mean_1.shape == log_std_1.shape == mean_2.shape == log_std_2.shape
    # ====
    # your code
    # define KL divergence of normal distributions

    return (
        log_std_2 - log_std_1 + \
              (torch.exp(2 * log_std_1) + (mean_1 - mean_2)** 2) / (2 * torch.exp(2 * log_std_2)) - 0.5
    ).sum()

    # ====


def main():
    print("Computing KL distances for ode paramters.")

    # load config files for data and pipeline
    with open("config.yaml", "r") as f1:
        config = yaml.full_load(f1)

    device = torch.device(config["device"])

    # dir with trajectories Datasets
    traj_dir = Path("trajectories/")
    if not traj_dir.exists():
        raise FileNotFoundError("No Dataset created for given activity.")

    # ode models
    models_dir = Path("models/")

    # compute mean and diag log std of the ML estimators
    model_distr = {}
    for model_file in models_dir.glob("*"):
        activity = model_file.stem

        print(f"Working with {activity}")

        vector_field = VectorFieldMLP(
            config["trajectory_dim"], 
            config["hidden_dim"]
        )
        ode_model = NeuralODE(vector_field, solver='rk4').to(device)
        ode_model.load_state_dict(
            torch.load(model_file)
        )

        # make dataloaders
        train_loader = DataLoader(
            torch.load(traj_dir / f"{activity}_train.pt"),
            config["batch_size"],
            shuffle=True
        )

        mean = vectorize_model_params(ode_model)
        log_std = fisher_hessian(ode_model, train_loader)

        model_distr[activity] = (mean, log_std)

    # compute pairwise KL for ML estimators
    pairwise_kl = defaultdict(list)
    for name_i, (mean_i, log_std_i) in model_distr.items():
        for name_j, (mean_j, log_std_j) in model_distr.items():
            pairwise_kl[name_i].append(
                get_normal_KL(
                    mean_i, log_std_i,
                    mean_j, log_std_j
                ).cpu().item()
            )

    # make KL Dataframe
    df_pairwise_kl = pd.DataFrame(
        pairwise_kl,
        index=list(model_distr.keys()),
        columns=list(model_distr.keys())
    )
    df_pairwise_kl.to_csv("pairwise_kl.csv", sep=" ")

    # make KL picture
    labels = list(df_pairwise_kl.columns)
    
    fig, ax = plt.subplots()
    mat_pic = ax.matshow(
        np.nan_to_num(np.log(df_pairwise_kl.values), neginf=0, nan=0)
    )
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_title("Activities log KL")
    fig.colorbar(mat_pic, ax=ax)
    fig.savefig("pairwise_kl.png")


if __name__ == "__main__":
    main()
