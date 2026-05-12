import os
from itertools import chain
from toolz import identity

import numpy as np
import torch
from torch import nn
from torch import optim
from lightning import LightningModule
from torchsde import sdeint


class SDEField(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, d: int):
        super().__init__()

        # self.A = nn.Linear(d, d, bias=False)
        self.A = nn.Linear(d, d)
        self.A.weight = nn.Parameter(1. * torch.randn_like(self.A.weight))
        self.nonlinear_add = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, d), nn.Tanh()
        )
        self.brownian_sigma = nn.Parameter(
            1e-1 * torch.randn((d, ))
        )
    
    def f(self, t: torch.Tensor, x: torch.Tensor):
        x = x.to(torch.float32)
        # return self.nonlinear_add(x)
        return self.A(x) + self.nonlinear_add(x)

    def g(self, t: torch.Tensor, x: torch.Tensor):
        return self.brownian_sigma.expand_as(x)


class FieldLitModule(LightningModule):
    def __init__(
        self, d: int, dt: float,
        state_names: list[str],
        traj_mean: torch.Tensor, traj_std: torch.Tensor
    ):
        super().__init__()
        self.field = SDEField(d)
        self.d = d
        self.dt = dt
        self.state_names = state_names
        self.register_buffer("traj_mean", traj_mean.to(torch.float32))
        self.register_buffer("traj_std", traj_std.to(torch.float32))

        self.save_hyperparameters(ignore=["traj_mean", "traj_std"])
        self.automatic_optimization = False
    
    def _rollout(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # batch: (B, T, d) -> (pred, target) both (T, B, d)
        batch = batch.to(torch.float32)
        T = batch.shape[1]
        x0 = batch[:, 0]
        t = torch.arange(T, dtype=torch.float32) * self.dt
        pred = sdeint(self.field, x0, t)
        target = batch.transpose(0, 1)

        return pred, target

    def training_step(self, batch, batch_idx):
        for opt in self.optimizers():
            opt.zero_grad()

        batch = (batch - self.traj_mean) / self.traj_std
        pred, target = self._rollout(batch)
        loss = nn.functional.mse_loss(pred, target)
        self.log("Train/loss", loss, on_step=True, on_epoch=True)

        self.manual_backward(loss)
        for opt in self.optimizers():
            opt.step()

    def validation_step(self, batch, batch_idx):
        batch = (batch - self.traj_mean) / self.traj_std
        pred, target = self._rollout(batch)
        loss = nn.functional.mse_loss(pred, target)
        self.log("Val/loss", loss, on_epoch=True)

        return loss
    
    def configure_optimizers(self):
        return [
            optim.Adam(
                chain(self.field.nonlinear_add.parameters()),
                lr=1e-3, weight_decay=1e-6
            ),
            optim.Adam(
                self.field.A.parameters(),
                lr=1e-3
            )
        ]
