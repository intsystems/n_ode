""" Script to launch datasets building.
    Operates in snakemake ecosystem.
"""
import re
import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from pipe import select

import torch
from torch.utils.data import random_split, TensorDataset

import wandb
from wandb.util import generate_id

from node.data_modules import build_subj_act_trajs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subj_id", type=int)
    parser.add_argument("data_config_path", type=Path)
    parser.add_argument("data_shared_config_path", type=Path)
    parser.add_argument("config_wandb_path", type=Path)
    parser.add_argument("raw_data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    wandb_config = OmegaConf.load(args.config_wandb_path)
    subj_id = args.subj_id
    # load config file
    data_config: DictConfig = OmegaConf.merge(
        OmegaConf.load(args.data_config_path),
        OmegaConf.load(args.data_shared_config_path)
    )

    torch.manual_seed(data_config.seed)

    run = wandb.init(
        name="data-" + generate_id(),
        config=data_config,
        tags=["normalized", f"subj{subj_id}", data_config.data.act],
        **dict(wandb_config)
    )

    # build trajectories
    sliced_trajs_list = build_subj_act_trajs(
        subj_id=subj_id,
        data_dir=args.raw_data_dir,
        **dict(data_config.data)
    )
    # concat trajectories
    sliced_trajs = {}
    tensor_names = ["traj", "dur", "subj_id", "traj_num"]
    for name in tensor_names:
        sliced_trajs[name] = torch.concat(list(
            sliced_trajs_list | select(lambda t: t[name])
        ))
    del sliced_trajs_list

    # normalize trajectories
    sliced_trajs["traj"] = \
        (sliced_trajs["traj"] - sliced_trajs["traj"].mean(dim=(0, 1))) \
            / sliced_trajs["traj"].std(dim=(0, 1))

    test_ratio = data_config.test_ratio
    train_indx, test_indx = random_split(
        range(sliced_trajs["traj"].shape[0]),
        [1 - test_ratio, test_ratio]
    )

    train_dataset = TensorDataset(*list(
        sliced_trajs.values() | select(lambda t: t[train_indx])
    ))
    test_dataset = TensorDataset(*list(
        sliced_trajs.values() | select(lambda t: t[test_indx])
    ))

    # save datasets
    torch.save(train_dataset, args.output_dir / "train.pkl")
    run.log_artifact(
        args.output_dir / "train.pkl",
        f"subj_{subj_id}_{data_config.data.act}_train",
        type="dataset"
    )
    torch.save(test_dataset, args.output_dir / "test.pkl")
    run.log_artifact(
        args.output_dir / "test.pkl",
        f"subj_{subj_id}_{data_config.data.act}_test",
        type="dataset"
    )
