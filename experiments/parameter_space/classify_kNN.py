""" Script to launch trajectories classification with kNN.
    Editable.
"""
import argparse
from pipe import select
from pathlib import Path
from omegaconf import OmegaConf
from rich.console import Console

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import wandb

from components.field_model import MyVectorField
from components.feature import get_acts_feat_splitted


console = Console()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints_dir", type=Path)
    parser.add_argument("split_config_path", type=Path)
    parser.add_argument("knn_config_path", type=Path)
    parser.add_argument(
        "shared_train_config_path", type=Path, 
        help="Needed to reconstuct vf and extract params"
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, required=False,
        help="parallelism for neigbours search"
    )
    args = parser.parse_args()

    knn_config = OmegaConf.load(args.knn_config_path)
    split_config = OmegaConf.load(args.split_config_path)
    train_config = OmegaConf.load(args.shared_train_config_path)

    acts_train_test = get_acts_feat_splitted(
        args.checkpoints_dir,
        MyVectorField(**dict(train_config.vf)),
        **dict(split_config)
    )
    # mapping act_name -> act_indx
    acts_to_indx = dict(zip(acts_train_test.keys(), range(len(acts_train_test))))
    # mapping act_indx -> act_name
    indx_to_act = dict(enumerate(acts_train_test.keys()))

    # do data transformations
    pass

    X_train = np.concat(list(
        acts_train_test.values() | select(lambda x: x["train"])
    ))
    X_test = np.concat(list(
        acts_train_test.values() | select(lambda x: x["val"])
    ))
    y_train = np.concat(list(
        iter(acts_train_test.items()) |
        select(lambda x: np.full(x[1]["train"].shape[0], acts_to_indx[x[0]]))
    ))

    classifier = KNeighborsClassifier(**dict(knn_config), n_jobs=args.num_workers)
    classifier.fit(X_train, y_train)

    y_pred = pd.Series(classifier.predict(X_test)).map(lambda v: indx_to_act[v])
    y_true = []
    for act in acts_to_indx:
        y_true.append(pd.Series(
            [act] * acts_train_test[act]["val"].shape[0]
        ))
    y_true = pd.concat(y_true, ignore_index=True)
    result = pd.DataFrame({
        "pred": y_pred,
        "true": y_true
    })
    result.to_csv("classify/kNN.csv", index=False)

    accuracy = (result["pred"] == result["true"]).mean()
    console.print("Accuracy = ", accuracy)
