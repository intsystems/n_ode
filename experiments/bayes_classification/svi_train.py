""" Training bayessian dynamical system
"""
from itertools import *
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from copy import deepcopy
import argparse

from pipe import select

import wandb

import numpy as np
import pandas as pd

import torch
from torch.utils.data import random_split, DataLoader, TensorDataset

from torchdiffeq import odeint_adjoint

import pyro
import pyro.distributions
import pyro.nn
import pyro.nn.module
import pyro.infer
import pyro.infer.autoguide

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.functional import mean_squared_error, mean_absolute_percentage_error, r2_score 

from bayes_modules import BayesNormDiagVectorField, BayessianNODE

from node.data_modules import ActivityDataModule, ActivityTrajDataset
from node.field_model import VectorFieldMLP


class LitBayessianNODE(L.LightningModule):
    def __init__(
        self,
        vf: torch.nn.Module,
        trajectory_len: int,
        trajectory_dim: int,
        dataset_size: int,
        optim_config: dict,
        odeint_kwargs: dict = {},
        num_particles: int = 1,
    ):
        super().__init__()

        self.dataset_size = dataset_size

        vf_ = deepcopy(vf)
        # create bayessian NODE model
        self.bayes_node = BayessianNODE(
            BayesNormDiagVectorField(vf_),
            trajectory_len,
            odeint_kwargs
        )
        # create ELBO Module with AutoNormal guide
        self.loss_fun = pyro.infer.Trace_ELBO(num_particles)(
            self.bayes_node,
            pyro.infer.autoguide.AutoNormal(self.bayes_node)
        )
        self.guide = self.loss_fun.guide

        # All relevant parameters need to be initialized before ``configure_optimizer`` is called.
        # Since we used AutoNormal guide our parameters have not been initialized yet.
        # Therefore we initialize the model and guide by running one mini-batch through the loss.
        self.loss_fun(
            dataset_size,
            duration=torch.tensor([trajectory_len] * 2),
            z_0=torch.ones((2, trajectory_dim))
        )

        self._predictor = pyro.infer.Predictive(self.bayes_node, guide=self.guide, num_samples=1)

        self.optim_config = optim_config

        # logging
        self.save_hyperparameters(ignore=["vf"], logger=False)

    def training_step(self, batch: tuple[torch.Tensor], batch_indx):
        traj, duration, subj = batch
        z_0 = traj[:, 0, :]

        elbo_loss = self.loss_fun(self.dataset_size, duration, z_0, traj)

        # logging
        self.log("Train/ELBO loss", elbo_loss, prog_bar=True, on_step=True)

        return elbo_loss
    
    def validation_step(self, batch: tuple[torch.Tensor], batch_indx):
        traj, duration, subj = batch
        z_0 = traj[:, 0, :]

        preds = self._predictor(self.dataset_size, duration, z_0)

        # logging
        self.log(
            "Test/RMSE",
            mean_squared_error(preds, traj, squared=False),
            on_epoch=True
        )
        self.log(
            "Test/MAPE",
            mean_absolute_percentage_error(preds, traj),
            on_epoch=True
        )
        self.log(
            "Test/R2",
            r2_score(preds, traj),
            on_epoch=True
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.loss_fun.parameters(), **self.optim_config)


def main(config: DictConfig):
    # create logger
    logger = WandbLogger(
        project="node",
        group="bayes",
        tags=["no-normalized", "dim20", "len500", "mlp_tanh", "jog_ups", "gyro_data"],
        config=config,
        mode="disabled" # debug
    )
    # turn str paths in config into Paths
    config.data = OmegaConf.create({
        k : (Path(v) if k in ("data_path", "save_dir") else v) for k, v in config.data.items()
    })

    # get dataset len to compute elbo correctly
    dataset_size = len(
        ActivityTrajDataset(*list(config.data.values())[:-2])
    )

    l_module = LitBayessianNODE(
        VectorFieldMLP(**config.vector_field),
        config.data.trajectory_len,
        config.data.trajectory_dim,
        dataset_size,
        config.optim,
        config.odeint,
        config.elbo.num_particles
    )

    act_data_module = ActivityDataModule(*config.data.values())

    trainer = L.Trainer(**config.trainer, logger=logger)
    trainer.fit(l_module, datamodule=act_data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.yaml",
                        required=False, help="Path to the configuration file.")
    args = parser.parse_args()

    # load config file
    config = OmegaConf.load(args.config_path)

    main(config)
