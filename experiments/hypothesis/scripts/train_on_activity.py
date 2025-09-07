""" Example script to launch model training.
    Editable.
"""
import argparse
import re
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from wandb.util import generate_id

from components.field_model import MyVectorField
from components.field_module import LitNodeHype


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_config", type=Path)
    parser.add_argument("data_config", type=Path)
    parser.add_argument("train_dataset", type=Path)
    parser.add_argument("test_dataset", type=Path)
    parser.add_argument("wandb_config", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    # load config files
    train_config: DictConfig = OmegaConf.load(args.train_config)
    data_config: DictConfig = OmegaConf.load(args.data_config)
    wandb_config: DictConfig = OmegaConf.load(args.wandb_config)

    subj_id = int(re.search(r"subj-(\d+)", wandb_config.group)[1])

    # set rand seed
    torch.manual_seed(train_config.seed)

    # choose vector field
    vf = MyVectorField(**dict(train_config.vf))
    loss_funcs = {
        "MSE": F.mse_loss,
        "MAE": F.l1_loss
    }
    lit_node = LitNodeHype(vf, loss_funcs, dict(train_config.optim), dict(train_config.odeint))

    logger = WandbLogger(
        name=f"train-{data_config.data.act}-" + generate_id(),
        tags=["mlp_tanh", "mse"],
        config=dict(train_config),
        log_model="all",
        # it only attributes to wandb cloud
        checkpoint_name=f"subj_{subj_id}_{data_config.data.act}_checkpoint",
        **dict(wandb_config)
    )

    # load train/test datasets
    train_dataset: TensorDataset = torch.load(args.train_dataset, weights_only=False)
    train_loader = DataLoader(train_dataset, train_config.batch_size, shuffle=True)
    test_dataset: TensorDataset = torch.load(args.test_dataset, weights_only=False)
    test_loader = DataLoader(test_dataset, train_config.batch_size, shuffle=False)
    logger.use_artifact(f"subj_{subj_id}_{data_config.data.act}_train:latest")
    logger.use_artifact(f"subj_{subj_id}_{data_config.data.act}_test:latest")

    earlystopping_callback = EarlyStopping(**dict(train_config.early_stop))
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename=f"{data_config.data.act}_checkpoint",
        # results will be overwritten locally
        enable_version_counter=False,
        **dict(train_config.model_checkpoint)
    )
    trainer = L.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, earlystopping_callback],
        **dict(train_config.trainer)
    )

    trainer.fit(lit_node, train_loader, test_loader)
