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
from torch.utils.data import random_split, DataLoader, TensorDataset, ConcatDataset

from node.train import compute_traj_batch_loss, predict_trajectory, train
from node.traj_build import make_trajectories, get_unit_dataset, normalize_traj
from train_config import get_model, get_optimizer, get_callbacks, get_eval_on_other_participants_callback


def main():
    # load config files for pipeline
    with open("train_config.yaml", "r") as f1:
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
        group="parameter_space",
        tags=["train", "no-normalized", "dim20", "len500", "linear_vec_field", "jog_ups"],
        config=config,
        #mode="disabled" # debug
    )

    print("Building phase trajectories...")
    make_trajectories(
        config,
        data_dir,
        traj_dir
    )

    # train models for each activity: activity_code, participant
    for act in data_params["activity_codes"]:
        for act_code in data_params["activity_codes"][act]:
            for partic in range(data_params["num_participants"]):
                print(f"Training {act}_{act_code}, participant {partic}...")
                train_dataset = get_unit_dataset(
                    act,
                    act_code,
                    partic,
                    traj_dir
                )

                # take same act but other act_codes for current participant for test
                other_act_codes = set(data_params["activity_codes"][act]).difference((act_code, ))
                test_dataset = []
                for other_act_code in other_act_codes:
                    test_dataset.append(
                        get_unit_dataset(
                            act,
                            other_act_code,
                            partic,
                            traj_dir
                        )
                    )
                test_dataset = ConcatDataset(test_dataset)

                # take several other random trajectories for current act, act_code
                # from other participants for test
                other_test_dataset = []
                rand_other_partic = np.random.choice(
                    list(set(range(data_params["num_participants"])) - set([partic])),
                    config["num_test_trajectories"]
                )
                for other_partic in rand_other_partic:
                    other_test_dataset.append(
                        get_unit_dataset(
                            act,
                            act_code,
                            other_partic,
                            traj_dir
                        )
                    )
                other_test_dataset = ConcatDataset(other_test_dataset)

                # normalize trajectories if neccessery
                # act_dataset = TensorDataset(normalize_trajectories(act_dataset.tensors[0]), act_dataset.tensors[1])

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
                other_test_loader = DataLoader(
                    other_test_dataset,
                    config["batch_size"],
                    shuffle=False
                )

                ode_model = get_model().to(device)
                optimizer = get_optimizer(ode_model, act)
                callbacks = get_callbacks(run, act, act_code, partic, test_loader, models_dir)
                # add evaluation on "other_test_dataset"
                callbacks["test"].append(
                    get_eval_on_other_participants_callback(
                        act,
                        act_code,
                        partic,
                        other_test_loader,
                        run
                    )
                )

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
