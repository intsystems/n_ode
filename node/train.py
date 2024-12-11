""" launches neural ode train processes for specific activity label
"""
from tqdm import tqdm
from typing import Callable

from matplotlib import pyplot as plt
from torchdyn.core import NeuralODE

import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import warnings
warnings.simplefilter("ignore", FutureWarning)


def get_trajectory_mask(durations: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
    mask = torch.ones_like(traj).to(traj.device)
    for i in range(mask.shape[0]):
        # mask out padding vectors in trajectory
        mask[i, durations[i]: , ...] = 0.

    return mask


def train_epoch(
    ode_model: NeuralODE,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    callback: Callable = None
):
    device = ode_model.device

    ode_model.train()
    for batch in tqdm(train_loader, desc="Train", leave=False):
        # debug
        break

        optimizer.zero_grad()

        traj: torch.Tensor = batch[0].to(device)
        durations: torch.Tensor = batch[1].to(device)
        
        traj_len = traj.shape[1]
        t_span = torch.arange(0, traj_len).to(device)
        mask = get_trajectory_mask(durations, traj)

        t_eval, traj_predict = ode_model(traj[:, 0, :], t_span)
        # move batch axis in front
        traj_predict = traj_predict.movedim(1, 0)

        # average loss among all real phase vectors
        loss = F.mse_loss(
            traj,
            traj_predict * mask,
            reduction="none"
        )
        # get l2-norm of traj. vectors residuals
        loss = loss.sum(dim=-1)
        # avarage on real traj. duration
        loss = loss.sum(dim=-1) / durations
        # mean across batch
        loss = loss.mean()

        loss.backward()
        optimizer.step()

        if callback is not None:
            callback(ode_model, {"mse": loss.item()})


@torch.no_grad
def eval_epoch(
    ode_model: NeuralODE,
    test_loader: DataLoader,
    callback: Callable = None
):
    device = ode_model.device

    ode_model.eval()
    # container for test batches losses
    test_losses = []
    for batch in tqdm(test_loader, desc="Test", leave=False):
        # debug
        break

        traj: torch.Tensor = batch[0].to(device)
        durations: torch.Tensor = batch[1].to(device)

        traj_len = traj.shape[1]
        t_span = torch.arange(0, traj_len).to(device)
        mask = get_trajectory_mask(durations, traj)

        t_eval, traj_predict = ode_model(traj[:, 0, :], t_span)
        # move batch axis in front
        traj_predict = traj_predict.movedim(1, 0)

        # average loss among all real phase vectors
        loss = F.mse_loss(
            traj,
            traj_predict * mask,
            reduction="none"
        )
        # get l2-norm of traj. vectors residuals
        loss = loss.sum(dim=-1)
        # avarage on real traj. duration
        loss = loss.sum(dim=-1) / durations
        # mean across batch
        loss = loss.mean()
        test_losses.append(loss)

    # debug
    test_average_loss = 0
    # test_average_loss = torch.stack(test_losses).mean().item()

    if callback is not None:
        callback(ode_model, {"mean_mse": test_average_loss})


def train(
    num_epochs: int,
    ode_model: NeuralODE,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: optim.Optimizer,
    callbacks: dict[Callable] = None
):
    # set default callbacks if not provided
    if callbacks is None:
        callbacks = {
            "pre_epoch": None,
            "train": None,
            "test": None,
            "post_epoch": None
        }

    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        # calling pre epoch callback
        # it can stop training
        if callbacks["pre_epoch"] is not None and callbacks["pre_epoch"](ode_model):
            print("Stopping early.")
            break

        train_epoch(
            ode_model,
            train_loader,
            optimizer,
            callbacks["train"]
        )

        eval_epoch(
            ode_model,
            test_loader,
            callbacks["test"]
        )

        # calling post epoch callbacks
        if callbacks["post_epoch"] is not None:
            callbacks["post_epoch"](ode_model)


@torch.no_grad
def vizualize_pred_traj(
    ode_model: NeuralODE,
    test_loader: DataLoader,
):
    device = ode_model.device

    # get first 3 test trajectories
    traj, durations = next(iter(test_loader))
    traj = traj[:3].to(device); durations = durations[:3].to(device)

    # get prediction
    traj_len = traj.shape[1]
    t_span = torch.arange(0, traj_len).to(device)
    _, traj_predict = ode_model(traj[:, 0, :], t_span)
    # move batch axis in front
    traj_predict = traj_predict.movedim(1, 0)

    fig, ax = plt.subplots(nrows=3, figsize=(12, 8))

    for i in range(3):
        # vizualize first two components of the trajectory vector
        true_traj = traj[i, durations[i]:, :2].cpu().numpy()
        pred_traj = traj_predict[i, :durations[i], :2].cpu().numpy()

        ax[i].scatter(true_traj[0][0], true_traj[0][1], color="red", markersize=20, label="start")
        ax[i].plot(
            *true_traj,
            label="true",
            marker="."
        )
        ax[i].plot(
            *pred_traj,
            label="pred",
            marker="^"
        )

        ax[i].set_title(f"Trajectory prediction {i}")
        ax[i].legend()
        ax[i].grid(True)

    return fig