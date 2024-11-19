""" computing KL-divergence between ML-estimators of diffrent activities
"""
import sys
from pathlib import Path
import yaml
from tqdm import tqdm
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torchdyn.core import NeuralODE
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from field_model import VectorField
from optimizer import get_optimizer
from train import get_trajectory_mask

import warnings
warnings.simplefilter("ignore", FutureWarning)


def vectorize_model_params(model: nn.Module) -> torch.Tensor:
    vec_params = []
    for param in model.parameters():
        vec_params.append(param.flatten())

    return torch.stack(vec_params).unsqueeze()


def fisher_hessian(ode_model: NeuralODE, train_loader: DataLoader) -> torch.Tensor:
    """ Computes diagonal elements of the Fisher matrix of log(p). It is factorized approximation
            of the ML inversed hessian. Returns diagonal log-std.
    """
    ode_model.zero_grad()
    ode_model.eval()

    num_batches = len(list(iter(train_loader)))

    for batch in tqdm(train_loader, desc="Batch", leave=False):
        traj: torch.Tensor = batch[0].to(ode_model.device)
        durations: torch.Tensor = batch[1].to(ode_model.device)
        
        traj_len = traj.shape[1]
        t_span = torch.arange(0, traj_len).to(ode_model.device)
        mask = get_trajectory_mask(durations, traj)

        t_eval, traj_predict = ode_model(traj[:, 0, :], t_span)
        # move batch axis in front
        traj_predict = traj_predict.movedim(1, 0)

        loss = F.mse_loss(
            traj.flatten(end_dim=-2), 
            (traj_predict * mask).flatten(end_dim=-2),
            reduction="sum"
        )
        loss /= durations.sum()
        loss /= num_batches

        # accumulate gradients
        loss.backward()

    # compute log std out of log(p) gradients
    diag_log_std = []
    for param in ode_model.parameters():
        diag_log_std.append(
            - torch.log(param.grad.flatten())
        )
    diag_log_std = torch.stack(diag_log_std).unsqueeze()

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
          log_std_2 - log_std_1 + (torch.exp(2 * log_std_1) + (mean_1 - mean_2) ** 2) / (2 * torch.exp(2 * log_std_2)) - 0.5
        )

    # ====


def main():
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

        vector_field = VectorField(config["trajectory_dim"], config["hidden_dim"])
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
    pairwise_kl = np.empty((len(model_distr), len(model_distr)))
    for i, mean_i, log_std_i in enumerate(model_distr.values()):
        for j, mean_j, log_std_j in enumerate(model_distr.values()):
            pairwise_kl[i][j] = get_normal_KL(
                mean_i, log_std_i,
                mean_j, log_std_j
            )

    # make KL Dataframe
    pd.DataFrame(
        pairwise_kl,
        index=list(model_distr.keys()),
        columns=list(model_distr.keys())
    ).to_csv("pairwise_kl.csv")

    # make KL picture
    fig, ax = plt.subplots()
    ax.matshow(pairwise_kl)
    ax.set_xticklabels(list(model_distr.keys()))
    ax.set_yticklabels(list(model_distr.keys()))
    fig.savefig("pairwise_kl.png")


if __name__ == "__main__":
    main()