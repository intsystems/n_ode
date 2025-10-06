""" Test data is the last activity code for each activity.
    Train data is the rest activity codes.
"""
import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from pipe import select

import torch
from torch.utils.data import TensorDataset


from node.data_modules import build_subj_act_trajs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subj_id", type=int)
    parser.add_argument("act", type=str)
    parser.add_argument("traj_len", type=int)
    parser.add_argument("data_config_path", type=Path)
    parser.add_argument("raw_data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    data_config: DictConfig = OmegaConf.load(args.data_config_path)

    torch.manual_seed(data_config.seed)

    # build trajectories
    sliced_trajs_list = build_subj_act_trajs(
        act=args.act,
        subj_id=args.subj_id,
        max_len=args.traj_len,
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

    # last trajectory goes to test, others - to train
    max_traj_num = sliced_trajs["traj_num"].max()
    split_index = (sliced_trajs["traj_num"] == max_traj_num).argmax().item()
    train_dataset = TensorDataset(*list(
        sliced_trajs.values() | select(lambda t: t[:split_index])
    ))
    test_dataset = TensorDataset(*list(
        sliced_trajs.values() | select(lambda t: t[split_index:])
    ))

    # save datasets
    torch.save(train_dataset, args.output_dir / "train.pkl")
    torch.save(test_dataset, args.output_dir / "test.pkl")
