from pathlib import Path
from pipe import select

import numpy as np

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

from .field_module import LitNodeSingleTraj


def make_feature_matrix(
    act_checkpoint_dir: Path,
    vf: nn.Module
) -> torch.Tensor:
    """make feature matrix out of activity vector fields
    """
    feat_matrix = []
    for ckpt in act_checkpoint_dir.glob("[!.]*"):
        vf = LitNodeSingleTraj.load_from_checkpoint(ckpt, vf=vf).vf
        feat_matrix.append(
            torch.concat(list(vf.parameters() | select(torch.flatten))).detach()
        )
    return torch.stack(feat_matrix)


def get_acts_feat_splitted(
    checkpoint_dir: Path,
    vf: nn.Module,
    test_size: float,
    seed: int
) -> dict[str, dict[np.ndarray]]:
    act_data_splitted = {}
    for act_checkpoint_dir in checkpoint_dir.glob("*"):
        act = act_checkpoint_dir.name
        feat_matrix = make_feature_matrix(act_checkpoint_dir, vf).numpy()
        feat_matrix_splitted = train_test_split(
            feat_matrix, test_size=test_size, random_state=seed
        )
        act_data_splitted[act] = {
            name: value
            for (name, value) in zip(["train", "val"], feat_matrix_splitted)
        }
    return act_data_splitted
