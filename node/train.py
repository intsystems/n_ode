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


def predict_trajectory(
    ode_model: NeuralODE,
    traj: torch.Tensor
) -> torch.Tensor:
    device = ode_model.device

    traj_len = traj.shape[1]
    t_span = torch.arange(0, traj_len).to(device)

    t_eval, traj_predict = ode_model(traj[:, 0, :], t_span)
    # move batch axis in front
    traj_predict = traj_predict.movedim(1, 0)

    return traj_predict


def compute_traj_batch_loss(
    traj_target: torch.Tensor,
    traj_predict: torch.Tensor,
    durations: torch.Tensor
) -> torch.Tensor:
    mask = get_trajectory_mask(durations, traj_target)

    # compute loss for each phase vector
    loss = F.mse_loss(
        traj_target,
        traj_predict * mask,
        reduction="none"
    )
    # get l2-norm of traj. vectors residuals
    loss = loss.sum(dim=-1)
    # avarage on real traj. duration
    loss = loss.sum(dim=-1) / durations

    return loss


def compute_traj_loss(
    traj_target: torch.Tensor,
    traj_predict: torch.Tensor,
    durations: torch.Tensor
) -> torch.Tensor:
    loss = compute_traj_batch_loss(traj_target, traj_predict, durations)
    # mean across batch
    loss = loss.mean()

    return loss


def train_epoch(
    epoch: int,
    ode_model: NeuralODE,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    callbacks: list[Callable] = None
):
    ode_model.train()
    for batch in tqdm(train_loader, desc="Train", leave=False):
        device = ode_model.device
        # debug
        # break

        optimizer.zero_grad()

        traj: torch.Tensor = batch[0].to(device)
        durations: torch.Tensor = batch[1].to(device)
        
        traj_predict = predict_trajectory(ode_model, traj)

        # average loss among all REAL phase vectors
        loss = compute_traj_loss(traj, traj_predict, durations)
        if torch.abs(loss) < 1e-3 or loss is None:
            pass

        loss.backward()
        optimizer.step()

        if callbacks is not None:
            for callback in callbacks:
                callback(epoch, ode_model, {"mse": loss.item()})


@torch.no_grad()
def eval_epoch(
    epoch: int,
    ode_model: NeuralODE,
    test_loader: DataLoader,
    callbacks: list[Callable] = None
):
    device = ode_model.device

    ode_model.eval()
    # container for test batches losses
    test_losses = []
    for batch in tqdm(test_loader, desc="Test", leave=False):
        # debug
        # break

        traj: torch.Tensor = batch[0].to(device)
        durations: torch.Tensor = batch[1].to(device)

        traj_predict = predict_trajectory(ode_model, traj)

        # average loss among all REAL phase vectors
        loss = compute_traj_loss(traj, traj_predict, durations)
        test_losses.append(loss)

    # debug
    # test_average_loss = 0
    test_average_loss = torch.stack(test_losses).mean().item()

    if callbacks is not None:
        for callback in callbacks:
                callback(epoch, ode_model, {"mean_mse": test_average_loss})


def train(
    num_epochs: int,
    ode_model: NeuralODE,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: optim.Optimizer,
    callbacks: dict[list[Callable]] = None
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
        if callbacks["pre_epoch"] is not None:
            stop_train = False
            for callback in callbacks["pre_epoch"]:
                stop_train |= callback(epoch, ode_model)
            
            if stop_train:
                print("Stopping early.")
                break

        train_epoch(
            epoch,
            ode_model,
            train_loader,
            optimizer,
            callbacks["train"]
        )

        eval_epoch(
            epoch,
            ode_model,
            test_loader,
            callbacks["test"]
        )

        # calling post epoch callbacks
        if callbacks["post_epoch"] is not None:
            for callback in callbacks["post_epoch"]:
                callback(epoch, ode_model)


@torch.no_grad()
def vizualize_pred_traj(
    ode_model: NeuralODE,
    test_loader: DataLoader,
):
    """ Makes matplotlib plot of predicited and true phase trajectories,
        projected on the first 2 dims
    """
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
