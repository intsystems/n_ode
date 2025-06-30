""" Script to extract features from vector fields and make common train-test split.
    Editable.
"""
import argparse
import pickle
from pathlib import Path
from omegaconf import OmegaConf
from rich.console import Console

import wandb
from wandb.util import generate_id

from components.field_model import MyVectorField
from components.feature import get_acts_feat_splitted

console = Console()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints_dir", type=Path)
    parser.add_argument("split_config_path", type=Path)
    parser.add_argument(
        "shared_train_config_path", type=Path, 
        help="Needed to reconstuct vf and extract params"
    )
    parser.add_argument("config_wandb_path", type=Path)
    parser.add_argument("out_features_file_name", type=str)
    args = parser.parse_args()

    split_config = OmegaConf.load(args.split_config_path)
    train_config = OmegaConf.load(args.shared_train_config_path)
    wandb_config = OmegaConf.load(args.config_wandb_path)

    run = wandb.init(
        name="featurize-" + generate_id(),
        **dict(wandb_config)
    )

    acts_train_test = get_acts_feat_splitted(
        args.checkpoints_dir,
        MyVectorField(**dict(train_config.vf)),
        **dict(split_config)
    )
    with open(args.out_features_file_name, "wb") as f:
        pickle.dump(acts_train_test, f)

    run.log_artifact(args.out_features_file_name, "act_features", type="dataset")
