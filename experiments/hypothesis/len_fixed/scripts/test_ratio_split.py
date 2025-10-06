import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from pipe import select

import wandb
from wandb.util import generate_id

import torch
from torch.utils.data import random_split, TensorDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("act", type=str)
    parser.add_argument("subj_id", type=int)
    parser.add_argument("data_config_path", type=Path)
    parser.add_argument("config_wandb_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    wandb_config = OmegaConf.load(args.config_wandb_path)
    # load config file
    data_config: DictConfig = OmegaConf.merge(
        OmegaConf.load(args.data_config_path),
        OmegaConf.load(args.data_shared_config_path)
    )

    torch.manual_seed(data_config.seed)

    run = wandb.init(
        name="data-" + generate_id(),
        config=data_config,
        tags=["normalized", f"subj{args.subj_id}", args.act,
              f"len{data_config.data.max_len}", "test_ratio_split"],
        **dict(wandb_config)
    )

    dataset = torch.load(args.output_dir / "full_dataset.pkl", weights_only=False)
    train_indx, test_indx = random_split(
        range(dataset["traj"].shape[0]),
        [1 - data_config.test_ratio, data_config.test_ratio]
    )
    train_dataset = TensorDataset(*list(
        dataset.values() | select(lambda t: t[train_indx])
    ))
    test_dataset = TensorDataset(*list(
        dataset.values() | select(lambda t: t[test_indx])
    ))
    # save datasets
    torch.save(train_dataset, args.output_dir / "train.pkl")
    run.log_artifact(
        args.output_dir / "train.pkl",
        f"subj_{args.subj_id}_{data_config.data.act}_train",
        type="dataset"
    )
    torch.save(test_dataset, args.output_dir / "test.pkl")
    run.log_artifact(
        args.output_dir / "test.pkl",
        f"subj_{args.subj_id}_{data_config.data.act}_test",
        type="dataset"
    )
