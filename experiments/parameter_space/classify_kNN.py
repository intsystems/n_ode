"""Experiment pipeline file
"""
from pathlib import Path
import yaml
from tqdm import tqdm

import wandb

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def vectorize_model_params(model: nn.Module) -> torch.Tensor:
    vec_params = []
    for param in model.parameters():
        vec_params.append(param.flatten().detach())

    return torch.concat(vec_params)


def main():
    # load config files for pipeline
    with open("kNN_config.yaml", "r") as f1:
        config = yaml.full_load(f1)
    with open("train_config.yaml", "r") as f1:
        train_config = yaml.full_load(f1)
    # load config files for data
    with open("../../data/dataset_params.yaml") as f2:
        data_params = yaml.full_load(f2)
    
    # dir for models
    models_dir = Path("models")
    if not models_dir.exists():
        raise FileExistsError("Models dir is not found.")

    # configure wandb run
    run = wandb.init(
        project="node",
        group="parameter_space",
        tags=["classify", "kNN", "jog_ups"],
        config=config
    )

    # create object-feature matrix and labels out of
    # model's params for classification
    X = []
    y = []
    for act_indx, act in enumerate(data_params["activity_codes"]):
        act_models = models_dir.glob(f"{act}_*")
        act_models = list(map(lambda file: vectorize_model_params(torch.load(file)).numpy(), act_models))
        X.extend(act_models)
        y.extend(np.repeat(act_indx, len(act_models)))
    X = np.concat(X)
    y = np.concat(y)

    # split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=train_config["test_size"], random_state=train_config["random_state"],
        stratify=list(range(len(data_params["activity_codes"])))
    )

    # fit/predict kNN classifier
    classifier = KNeighborsClassifier(**config)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # calc recall for each activity
    for act_indx, act in enumerate(data_params["activity_codes"]):
        act_labels = (y_test == act_indx)

        run.log(
            {
                f"Test/Accuracy_{act}": (y_pred[act_labels] == act_indx).mean()
            },
            commit=False
        )
    # calc overall accuracy
    run.log(
        {
            f"Test/Accuracy": accuracy_score(y_test, y_pred)
        },
        commit=True
    )


if __name__ == "__main__":
    main()