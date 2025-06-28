""" Script to launch trajectories classification with kNN.
    Editable.
"""
import argparse
import pickle
from pipe import select
from pathlib import Path
from omegaconf import OmegaConf
from rich.console import Console

import numpy as np
import pandas as pd

import xgboost as xgb

import wandb

from components.field_model import MyVectorField
from components.feature import get_acts_feat_splitted


console = Console()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("act_feat_matr_path", type=Path)
    parser.add_argument("boost_config_path", type=Path)
    parser.add_argument("model_save_dir", type=Path)
    parser.add_argument("classify_save_dir", type=Path)
    args = parser.parse_args()

    boost_config = OmegaConf.load(args.boost_config_path)

    with open(args.act_feat_matr_path, "rb") as f:
        acts_train_test = pickle.load(f)
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
    y_test = np.concat(list(
        iter(acts_train_test.items()) |
        select(lambda x: np.full(x[1]["val"].shape[0], acts_to_indx[x[0]]))
    ))

    classifier = xgb.XGBClassifier(
        **dict(boost_config)
    )
    with console.status("Training model..."):
        classifier.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    with open(args.model_save_dir / "boost.pkl", "wb") as f:
        pickle.dump(classifier, f)

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
    save_dir = Path(args.classify_save_dir)
    result.to_csv(save_dir / "boost.csv", index=False)
