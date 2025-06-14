""" Example script to launch models training.
    Editable.
"""
from pathlib import Path
from snakemake.script import snakemake
from omegaconf import OmegaConf, DictConfig

import torch
from torch.utils.data import TensorDataset

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from node.field_model import VectorFieldMLP
from node.data_modules import ActivityDataModule
from components.field_module import LitNodeHype


def process_subjects(proc_id: int, subj_list: list[int]):
    # load config files
    train_config: DictConfig = OmegaConf.load(snakemake.input["train_config"])
    data_config: DictConfig = OmegaConf.load(snakemake.input["data_config"])
    wandb_config: DictConfig = OmegaConf.load(snakemake.input["wandb_config"])
    
    # logging only on the part of data
    if proc_id == 0:
        logger = WandbLogger(
            tags=["train", "unnormalized", "mlp_tanh"],
            config=dict(train_config) | dict(data_config),
            log_model="all",
            # it only attributes to wandb cloud
            checkpoint_name=f"{data_config.data.act}_checkpoint",
            # mode="disabled", # debug
            **dict(wandb_config)
        )

    for subj_id in subj_list:
        for traj_dir in (Path(snakemake.input["dataset_dir"]) / f"subj_{subj_id}").glob("*/"):
            file_names = ["traj.pt", "dur.pt", "subj_id.pt", "traj_num.pt"]
            train_dataset = TensorDataset(*[
                torch.load(traj_dir / name, weights_only=True) for name in file_names
            ])
            
            # choose other random trajectory for validation
            ...



if __name__ == "__main__":
    # load config files
    train_config: DictConfig = OmegaConf.load(snakemake.input["train_config"])
    data_config: DictConfig = OmegaConf.load(snakemake.input["data_config"])
    wandb_config: DictConfig = OmegaConf.load(snakemake.input["wandb_config"])

    # set rand seed
    torch.manual_seed(train_config.seed)

    # choose vector field
    vf = VectorFieldMLP(**dict(train_config.vf))
    lit_node = LitNodeHype(vf, dict(train_config.optim), dict(train_config.odeint))

    logger = WandbLogger(
        tags=["train", "unnormalized", "mlp_tanh"],
        config=dict(train_config) | dict(data_config),
        log_model="all",
        # it only attributes to wandb cloud
        checkpoint_name=f"{data_config.data.act}_checkpoint",
        # mode="disabled", # debug
        **dict(wandb_config)
    )

    earlystopping_callback = EarlyStopping(
        monitor="Val/MSE",
        mode="min",
        min_delta=1e-3,
        patience=15
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"{data_config.data.act}_checkpoint",
        # results will be overwritten locally
        enable_version_counter=False,
        monitor="Val/MSE",
        mode="min"
    )
    trainer = L.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, earlystopping_callback],
        **dict(train_config.trainer)
    )

    trainer.fit(lit_node, datamodule=activity_data)
    print(checkpoint_callback.best_model_path)