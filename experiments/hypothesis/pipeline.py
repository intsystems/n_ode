"""Experiment pipeline file
"""
from collections import defaultdict
from pathlib import Path
import yaml
from tqdm import tqdm

import wandb

import numpy as np
import pandas as pd

import torch
from torch.utils.data import random_split, DataLoader, TensorDataset

from node.train import compute_traj_batch_loss, predict_trajectory, train
from node.traj_build import make_trajectories, make_activity_dataset, normalize_traj
from train_config import get_model, get_optimizer, get_callbacks


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
        group="hypothesis",
        tags=["no-normalized", "dim20", "len500", "mlp_tanh", "jog_ups"],
        config=config
    )

    print("Building phase trajectories...")
    make_trajectories(
        config,
        data_dir,
        traj_dir
    )
    
    test_loaders = {}
    ode_models = {}

    # train models for each activity
    print("Training models...")
    for act in data_params["activity_codes"]:
        act_dataset = make_activity_dataset(act, traj_dir)
        # normalize trajectories
        # act_dataset = TensorDataset(normalize_trajectories(act_dataset.tensors[0]), act_dataset.tensors[1])

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
        # save test trajectories for futher classification
        test_loaders[act] = test_loader

        ode_model = get_model().to(device)
        optimizer = get_optimizer(ode_model, act)
        callbacks = get_callbacks(run, act, test_loader, models_dir)

        print(f"Training {act}...")
        if act == "ups":
            num_epochs = config["num_epochs"]
        else:
            # use more epochs for jog
            num_epochs = config["num_epochs"] + 4

        train(  
            num_epochs,
            ode_model,
            train_loader,
            test_loader,
            optimizer,
            callbacks
        )

        # save model for futher classification
        ode_models[act] = ode_model

    # classify tests trajectories with trained models
    # using bayessian statistical testing
    # assume all activities are equally probable
    # then label = argmax of liklyhoods
    print("Classifying test trajectories...")
    for act, test_loader in test_loaders.items():
        print(f"Computing test losses for {act}")
        act_loss = defaultdict(lambda: torch.empty((0, ), device=device))

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test", leave=False):
                device = ode_model.device

                traj: torch.Tensor = batch[0].to(device)
                durations: torch.Tensor = batch[1].to(device)

                # compute batch loss for all models
                for ode_model_act, ode_model in ode_models.items():
                    traj_predict = predict_trajectory(ode_model, traj)
                    # average loss among all REAL phase vectors
                    loss_batch = compute_traj_batch_loss(traj, traj_predict, durations)

                    act_loss[ode_model_act] = torch.concat(
                        [act_loss[ode_model_act], loss_batch],
                        dim=0
                    )

                # debug
                # break

        # transform results to Dataframe
        act_loss = {
            act: loss_torch.cpu().numpy() for act, loss_torch in act_loss.items()
        }
        act_loss = pd.DataFrame(data=act_loss)
        # log lh table
        run.log({f"losses_{act}": wandb.Table(dataframe=act_loss)})

        # compute activity pedictions
        act_pred = act_loss.idxmin(axis="columns")
        accuracy = (act_pred == act).mean()
        # log accuracy for current activity
        run.log({f"accuracy_{act}": accuracy})


if __name__ == "__main__":
    main()
