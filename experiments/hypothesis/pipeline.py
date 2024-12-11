"""Experiment pipeline file
"""
from pathlib import Path
import yaml

import wandb

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from node.traj_build import make_trajectories, make_activity_dataset
from train import get_model, get_optimizer, get_callbacks
from node.train import train


def main():
    # load config files for pipeline
    with open("config.yaml", "r") as f1:
        config = yaml.full_load(f1)
    # load config files for data
    with open("../../data/dataset_params.yaml") as f2:
        data_params = yaml.full_load(f2)
    data_dir = Path("../../data")
    traj_dir = Path(config["traj_dir"])
    
    # create dir for models
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    device = torch.device(config["device"])

    # configure wandb run
    run = wandb.init(
        project="node",
        tags=["hypothesis", "no_normalize", "dim20", "mlp_field"]
    )

    make_trajectories(
        config,
        data_dir,
        traj_dir
    )
    
    # train models for each activity
    for act in data_params["activity_codes"]:
        act_dataset = make_activity_dataset(act, traj_dir)
        train_dataset, test_dataset = random_split(
            act_dataset,
            [1 - config["test_ratio"], config["test_ratio"]]
        )
        train_loader = DataLoader(
            train_dataset,
            config["batch_size"],
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset,
            config["batch_size"],
            shuffle=False
        )

        ode_model = get_model().to(device)
        optimizer = get_optimizer(ode_model, act)
        callbacks = get_callbacks(run, act, test_loader, models_dir)

        train(
            config["num_epochs"],
            ode_model,
            train_loader,
            test_loader,
            optimizer,
            callbacks
        )


if __name__ == "__main__":
    main()
