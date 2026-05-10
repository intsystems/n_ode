import os
from itertools import chain
from toolz import identity

import numpy as np
import torch
from torch import nn
from torch import optim
from lightning import LightningModule
# from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

import plotly.graph_objects as go

from experiment.motion_sense.utils.metric import SmoothedTrajectoryLoss


class Field(nn.Module):
    def __init__(self, d: int):
        super().__init__()

        self.A = nn.Linear(d, d, bias=False)
        INIT_SCALE = 1.
        # initalize small
        self.A.weight = nn.Parameter(INIT_SCALE * torch.randn_like(self.A.weight))

        self.nonlinear_add = nn.Sequential(
            nn.BatchNorm1d(d), nn.Linear(d, d), nn.ReLU(),
            nn.BatchNorm1d(d), nn.Linear(d, d), nn.ReLU(),
            nn.BatchNorm1d(d), nn.Linear(d, d), nn.ReLU(),
            nn.BatchNorm1d(d), nn.Linear(d, d), nn.ReLU(),
            nn.BatchNorm1d(d), nn.Linear(d, d)
        )
    
    def forward(self, t: torch.Tensor, x: torch.Tensor):
        x = x.to(torch.float32)
        return self.A(x) + self.nonlinear_add(x)


class FieldLitModule(LightningModule):
    def __init__(
        self, d: int, dt: float,
        state_noise_sigma: np.ndarray, noise_sigma: np.ndarray,
        state_names: list[str],
        traj_mean: torch.Tensor, traj_std: torch.Tensor
    ):
        super().__init__()
        self.field = Field(d)
        self.d = d
        self.dt = dt
        self.state_names = state_names
        self.register_buffer("traj_mean", traj_mean.to(torch.float32))
        self.register_buffer("traj_std", traj_std.to(torch.float32))

        kalman = UnscentedKalmanFilter(
            d, d, dt, hx=identity, fx=self._kalman_f,
            points=MerweScaledSigmaPoints(d, alpha=.1, beta=2., kappa=-1)
        )
        np.fill_diagonal(kalman.Q, state_noise_sigma ** 2)
        noise_sigma = noise_sigma / traj_std.numpy()
        np.fill_diagonal(kalman.R, noise_sigma ** 2)
        MAX_LEN = 100
        self.traj_smooth_loss = SmoothedTrajectoryLoss(kalman, MAX_LEN)

        self.save_hyperparameters(ignore=["traj_mean", "traj_std"])
        self.automatic_optimization = False
    
    @torch.no_grad
    def _kalman_f(self, x, dt):
        """Field for FilterField
        """
        x = torch.from_numpy(x)[None, ...]
        return odeint(
            self.field, x, torch.tensor([0., dt])
        )[-1].numpy().flatten()
    
    def _rollout(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # batch: (B, T, d) -> (pred, target) both (T, B, d)
        batch = batch.to(torch.float32)
        T = batch.shape[1]
        x0 = batch[:, 0]
        t = torch.arange(T, dtype=torch.float32) * self.dt
        pred = odeint(self.field, x0, t)
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

        self.traj_smooth_loss.update(batch[:, 0])

        return loss

    def on_validation_epoch_end(self):    
        loss, traj, traj_smooth = self.traj_smooth_loss.compute()
        self.log(
            "Val/smooth_rmse_loss", loss.item()
        )
        self.traj_smooth_loss.reset()

        for i, state_name in enumerate(self.state_names):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=np.arange(traj.shape[0]) * self.dt, 
                y=traj[:, i],
                mode='lines',
                name="traj"
            ))
            fig.add_trace(go.Scatter(
                x=np.arange(traj.shape[0]) * self.dt, 
                y=traj_smooth[:, i],
                mode='lines',
                name="smooth"
            ))
            fig.update_layout(
                xaxis_title="t", yaxis_title=state_name
            )
            self.logger.experiment.log_figure(
                self.logger.run_id,
                fig, os.path.join(state_name, f"{self.current_epoch}.html")
            )
    
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
