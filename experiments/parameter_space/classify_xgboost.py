""" Script to launch trajectories classification with gaussian bayes.
    Editable.
"""
import argparse
import re
from pipe import select
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from rich.console import Console

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import wandb

import torch
from torch.utils.data import TensorDataset, DataLoader

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from node.field_model import VectorFieldLinear
from components.field_module import LitNodeSingleTraj
from components.feature import make_feature_matrix


console = Console()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints_dir", type=Path)
    parser.add_argument("split_config_path", type=Path)
    parser.add_argument("gauss_config_path", type=Path)
    args = parser.parse_args()

    gauss_config = OmegaConf.load(args.gauss_config_path)
    split_config = OmegaConf.load(args.split_config_path)

    acts_ckpt_dir: dict[str, Path] = {
        ckpt_dir.name: ckpt_dir
        for ckpt_dir in args.checkpoints_dir.glob("*")
    }
    act_to_indx = {act: i for i, act in enumerate(acts_ckpt_dir)}
    indx_to_act = {i: act for i, act in enumerate(acts_ckpt_dir)}

    acts_feat_matrix = {
        act: make_feature_matrix(act_dir).numpy()
        for act, act_dir in acts_ckpt_dir
    }
    acts_labels = {
        act: np.full(act_feat_matrix.shape[0], act_to_indx[act])
        for act, act_feat_matrix in acts_feat_matrix
    }
    acts_train_test = {
        act: train_test_split(acts_feat_matrix[act], acts_labels[act], **dict(split_config))
        for act in act_to_indx
    }

    # do data transformations
    pass

    classifiers = {
        act: GaussianMixture(**dict(gauss_config))
        for act in act_to_indx
    }
    for act, classifier in classifiers.items():
        classifier.fit(*acts_train_test[act][:2])
    
    # assume each class has equal prior prob
    result = []
    for act in act_to_indx:
        cur_X_test = acts_train_test[act][3]
        acts_probs = {
            pred_act: classifier.predict_proba(cur_X_test).dot(classifier.weights_)
            for pred_act, classifier in classifiers.items()
        }
        acts_probs = pd.DataFrame(acts_probs)
        preds = acts_probs.idxmax(axis=1)
        result.append(pd.DataFrame({
            "pred": preds,
            "true": [act] * preds.size
        }))
    result = pd.concat(result)
    result.to_csv("classify/gauss.csv", index=False)

    accuracy = (result["pred"] == result["true"]).mean()
    console.print("Accuracy = ", accuracy)
