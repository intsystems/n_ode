""" Script to launch models training.
    Editable.
"""
import sys
import os
import argparse
import multiprocessing as mp
import re
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from rich.console import Console

import torch
from torch.utils.data import TensorDataset, DataLoader

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from components.field_model import MyVectorField
from components.field_module import LitNodeSingleTraj


def process_subjects(
    args: argparse.Namespace,
    proc_id: int,
    subj_list: list[int]
):
    # load config files
    train_config: DictConfig = OmegaConf.merge(
        OmegaConf.load(args.config_train_path),
        OmegaConf.load(args.config_shared_train_path)
    )
    data_config: DictConfig = OmegaConf.merge(
        OmegaConf.load(args.config_data_path),
        OmegaConf.load(args.config_shared_data_path)
    )
    wandb_config: DictConfig = OmegaConf.load(args.config_wandb_path)

    # set rand seed
    torch.manual_seed(train_config.seed)
    
    # logging only on the part of data
    # does not save models to wandb
    if proc_id == 0:
        logger = WandbLogger(
            tags=["train", "unnormalized", "mlp_tanh"],
            config=dict(train_config) | dict(data_config),
            mode="disabled", # debug
            **dict(wandb_config)
        )
        console = Console()
    else:
        logger = None
        console = None
        # disable Trainer logs
        f = open(os.devnull, 'w')
        sys.stdout = f
        sys.stderr = f

    data_file_names = ["traj.pt", "dur.pt", "subj_id.pt", "traj_num.pt"]

    # fix validation trajectory
    val_subj_id = subj_list[0]
    val_data_dir = next(
        (args.dataset_dir / f"subj_{val_subj_id}").glob("*")
    )
    val_dataset = TensorDataset(*[
        torch.load(val_data_dir / name, weights_only=True) for name in data_file_names
    ])
    val_traj_num = int(val_data_dir.name[5:])
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    for subj_id in subj_list:
        for traj_dir in (args.dataset_dir / f"subj_{subj_id}").glob("*/"):
            train_dataset = TensorDataset(*[
                torch.load(traj_dir / name, weights_only=True) for name in data_file_names
            ])
            train_traj_num = int(traj_dir.name[5:])
            train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

            if console is not None:
                console.log(f"Training on trajectory {train_traj_num}")

            vf = MyVectorField(**dict(train_config.vf))
            lit_node = LitNodeSingleTraj(
                vf,
                dict(train_config.optim),
                dict(train_config.odeint),
                train_traj_num,
                val_traj_num
            )

            checkpoint_callback = ModelCheckpoint(
                dirpath=f"tmp/checkpoints/{data_config.data.act}",
                filename=f"traj_{train_traj_num}",
                # results will be overwritten locally
                enable_version_counter=False,
                monitor=f"traj_{train_traj_num}/Val/MSE",
                mode="min"
            )
            
            trainer = L.Trainer(
                logger=logger,
                callbacks=[checkpoint_callback],
                **dict(train_config.trainer)
            )
            trainer.fit(lit_node, train_dataloader, val_dataloader)


def get_subj_list(dataset_dir: Path):
    subj_list = []
    for subj_dir in dataset_dir.glob("subj_*"):
        subj_list.append(int(re.fullmatch(r"subj_(\d+)", subj_dir.name)[1]))
    return subj_list


def devide_subj_list(subj_list: list, num_workers: int):
    subj_per_worker = len(subj_list) // num_workers
    residue = len(subj_list) % num_workers
    subj_list_divided = []

    cur_start = 0
    for i in range(num_workers):
        cur_finish = cur_start + subj_per_worker
        if residue > 0:
            cur_finish += 1
        subj_list_divided.append(subj_list[cur_start: cur_finish])
        residue -= 1
        cur_start = cur_finish

    return subj_list_divided


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("config_train_path", type=Path)
    parser.add_argument("config_shared_train_path", type=Path)
    parser.add_argument("config_data_path", type=Path)
    parser.add_argument("config_shared_data_path", type=Path)
    parser.add_argument("config_wandb_path", type=Path)
    parser.add_argument("num_workers", type=int)
    args = parser.parse_args()

    mp.set_start_method("spawn")

    # get available subjects and distribute among workers
    subj_list = get_subj_list(args.dataset_dir)
    subj_list_divided = devide_subj_list(subj_list, args.num_workers)
    
    proc_list = []
    for proc_indx in range(1, args.num_workers):
        cur_proc = mp.Process(target=process_subjects, args=(args, proc_indx, subj_list_divided[proc_indx]))
        cur_proc.start()
        proc_list.append(cur_proc)
    
    # main process also does work
    process_subjects(args, 0, subj_list_divided[0])

    for proc in proc_list:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError("Worker subprocess error!")
