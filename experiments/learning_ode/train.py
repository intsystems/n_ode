""" launches neural ode train processes for specific activity label
"""
import sys
from pathlib import Path
import yaml
from tqdm import tqdm
from datetime import datetime

import torch
from torch import nn
from torch.nn import functional as F
from torchdyn.core import NeuralODE
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from field_model import *
from optimizer import get_optimizer

import warnings
warnings.simplefilter("ignore", FutureWarning)


def get_trajectory_mask(durations: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
    mask = torch.ones_like(traj).to(traj.device)
    for i in range(mask.shape[0]):
        # mask out padding vectors in trajectory
        mask[i, durations[i]: , ...] = 0.

    return mask


def main(activity: str):
    print(f"Learning {activity} neural ode.")

    # load config files for data and pipeline
    with open("config.yaml", "r") as f1:
        config = yaml.full_load(f1)

    # ode model dir
    models_dir = Path("models/")
    models_dir.mkdir(exist_ok=True)

    device = torch.device(config["device"])

    # dir with trajectories Datasets
    traj_dir = Path("trajectories/")
    if not traj_dir.exists():
        raise FileNotFoundError("No Dataset created for given activity.")
    # make dataloaders
    train_loader = DataLoader(
        torch.load(traj_dir / f"{activity}_train.pt"),
        config["batch_size"],
        shuffle=True
    )
    test_loader = DataLoader(
        torch.load(traj_dir / f"{activity}_test.pt"),
        config["batch_size"],
        shuffle=False
    )

    # create model
    vector_field = eval(config["model_cls"])(config["trajectory_dim"], **config["model_params"])
    ode_model = NeuralODE(vector_field, solver='rk4').to(device)

    # get optimizer
    optim = get_optimizer(ode_model)

    # metrics writer
    writer = SummaryWriter(f"runs/{activity}/{datetime.now()}")

    # train loop
    global_step = 0
    for epoch in tqdm(range(config["num_epochs"]), desc="Epoch"):
        ode_model.train()
        for batch in tqdm(train_loader, desc="Train", leave=False):
            optim.zero_grad()

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
                traj.flatten(end_dim=-2),
                (traj_predict * mask).flatten(end_dim=-2),
                reduction="none"
            )
            # try weightening
            # loss *= 1 / torch.sqrt(torch.arange(start=1, end=loss.shape[0] + 1).unsqueeze(1).to(device))
            loss = (loss.sum(dim=1) / durations).sum()
            loss.backward()
            optim.step()

            writer.add_scalar("Train/MSE", loss.item(), global_step)
            global_step += 1

        ode_model.eval()
        with torch.no_grad():
            losses = []
            for batch in tqdm(test_loader, desc="Test", leave=False):
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
                    traj.flatten(end_dim=-2),
                    (traj_predict * mask).flatten(end_dim=-2),
                    reduction="none"
                )
                # try weightening
                # loss *= 1 / torch.sqrt(torch.arange(start=1, end=loss.shape[0] + 1).unsqueeze(1).to(device))
                loss = (loss.sum(dim=1) / durations).sum()
                losses.append(loss)

            writer.add_scalar("Test/MSE", torch.stack(losses).mean().item(), epoch)

            # save the ode model
            torch.save(ode_model.state_dict(), models_dir / f"{activity}.pt")

    writer.close()

if __name__ == "__main__":
    # read activity label from input
    activity: str = sys.argv[1]

    main(activity)
