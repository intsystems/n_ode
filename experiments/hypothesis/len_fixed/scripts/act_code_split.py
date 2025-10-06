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
    data_config: DictConfig = OmegaConf.load(args.data_config_path)

    torch.manual_seed(data_config.seed)

    run = wandb.init(
        name="data-" + generate_id(),
        config=data_config,
        tags=["normalized", f"subj{args.subj_id}", args.act,
              f"len{data_config.data.max_len}", "act_code_split"],
        **dict(wandb_config)
    )

    dataset: TensorDataset = torch.load(args.output_dir / "full_dataset.pkl", weights_only=False)
    # last trajectory goes to test, others - to train
    traj_num_tensor = dataset.tensors[3]
    max_traj_num = traj_num_tensor.max()
    split_index = (traj_num_tensor == max_traj_num).int().argmax().item()
    train_dataset = TensorDataset(*list(
        dataset.tensors | select(lambda t: t[:split_index])
    ))
    test_dataset = TensorDataset(*list(
        dataset.tensors | select(lambda t: t[split_index:])
    ))
    # save datasets
    torch.save(train_dataset, args.output_dir / "train.pkl")
    run.log_artifact(
        args.output_dir / "train.pkl",
        f"subj_{args.subj_id}_{args.act}_train",
        type="dataset"
    )
    torch.save(test_dataset, args.output_dir / "test.pkl")
    run.log_artifact(
        args.output_dir / "test.pkl",
        f"subj_{args.subj_id}_{args.act}_test",
        type="dataset"
    )

    run.log({f"train_datapoints_{args.act}": train_dataset.tensors[0].numel()})
