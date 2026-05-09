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
        INIT_SCALE = 1e-3
        # initalize small
        self.A.weight = nn.Parameter(INIT_SCALE * torch.randn_like(self.A.weight))

        self.nonlinear_add_scale = nn.Parameter(1e-3 * torch.randn((1,)))
        self.nonlinear_add = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, d), nn.Tanh()
        )
    
    def forward(self, t: torch.Tensor, x: torch.Tensor):
        x = x.to(torch.float32)
        return self.A(x) + self.nonlinear_add_scale * self.nonlinear_add(x)


class FieldLitModule(LightningModule):
    def __init__(
        self, d: int, dt: float, 
        state_noise_sigma: np.ndarray, noise_sigma: float,
        state_names: list[str]
    ):
        super().__init__()
        self.field = Field(d)
        self.d = d
        self.dt = dt
        self.state_names = state_names

        kalman = UnscentedKalmanFilter(
            d, d, dt, hx=identity, fx=self._kalman_f,
            points=MerweScaledSigmaPoints(d, alpha=.1, beta=2., kappa=-1)
        )
        kalman.R *= noise_sigma ** 2
        np.fill_diagonal(kalman.Q, state_noise_sigma ** 2)
        self.traj_smooth_loss = SmoothedTrajectoryLoss(kalman)

        self.save_hyperparameters()
        self.automatic_optimization = False
    
    @torch.no_grad
    def _kalman_f(self, x, dt):
        """Field for FilterField
        """
        x = torch.from_numpy(x)
        return odeint(
            self.field, x, torch.tensor([0., dt])
        )[-1].numpy()
    
    def training_step(self, batch, batch_idx):
        for opt in self.optimizers():
            opt.zero_grad()

        x0 = batch[:, :self.d]
        x_next = batch[:, self.d:]
        x_next_pred = odeint(
            self.field, x0, torch.tensor([0., self.dt]),
            options={"dtype": torch.float32}
        )[-1]
        loss = nn.functional.mse_loss(x_next_pred, x_next)
        self.log("Train/loss", loss, on_step=True, on_epoch=True)

        self.manual_backward(loss)
        for opt in self.optimizers():
            opt.step()
        # return loss
    
    def validation_step(self, batch, batch_idx):
        x0 = batch[:, :self.d]
        x_next = batch[:, self.d:]
        x_next_pred = odeint(
            self.field, x0, torch.tensor([0., self.dt]),
            options={"dtype": torch.float32}
        )[-1]
        loss = nn.functional.mse_loss(x_next_pred, x_next)
        self.log("Val/loss", loss, on_epoch=True)

        self.traj_smooth_loss.update(x0)

        return loss

    def on_validation_epoch_end(self):
        loss, traj, traj_smooth = self.traj_smooth_loss.compute()
        self.log("Val/smooth_rmse_loss", loss)
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
                chain(self.field.nonlinear_add.parameters(), [self.field.nonlinear_add_scale]),
                lr=1e-5, weight_decay=1e-5
            ),
            optim.Adam(
                self.field.A.parameters(),
                lr=1e-5
            )
        ]
