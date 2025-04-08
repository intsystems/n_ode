from typing import Callable
from pipe import select

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from torchdiffeq import odeint_adjoint

import lightning as L


def get_trajectory_mask(durations: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
    mask = torch.ones_like(traj).to(traj.device)
    for i in range(mask.shape[0]):
        # mask out padding vectors in trajectory
        mask[i, durations[i]: , ...] = 0.

    return mask


class LitNode(L.LightningModule):
    def __init__(
        self,
        vf: nn.Module,
        optim_kwargs: dict,
        odeint_kwargs: dict
    ):
        super().__init__()

        self.vf = vf
        self.optim_kwargs = optim_kwargs
        self.odeint_kwargs = odeint_kwargs

        self.save_hyperparameters(ignore=["vf"], logger=False)

    def training_step(self, batch, batch_idx):
        traj, duration, subj, traj_num = batch
        z_0 = traj[:, 0, :]

        pred = self.forward(z_0, num_steps=traj.shape[1])
        mask = get_trajectory_mask(duration, traj)
        
        # compute mean l2-loss for trajectories
        l2_loss, _ = self._compute_abs_rel_metric(
            F.mse_loss,
            traj,
            duration,
            pred,
            mask
        )

        return l2_loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        traj, duration, subj, traj_num = batch
        z_0 = traj[:, 0, :]

        pred = self.forward(z_0, num_steps=traj.shape[1])
        mask = get_trajectory_mask(duration, traj)
        
        # compute l2-loss
        l2_loss, l2_loss_rel = self._compute_abs_rel_metric(
            F.mse_loss,
            traj,
            duration,
            pred,
            mask
        )
        self.log("Train/MSE", l2_loss, prog_bar=True, on_step=True)
        self.log("Train/relMSE", l2_loss_rel, on_step=True)
        # compute l1-loss
        l1_loss, l1_loss_rel = self._compute_abs_rel_metric(
            F.l1_loss,
            traj,
            duration,
            pred,
            mask
        )
        self.log("Train/MAE", l1_loss, on_step=True)
        self.log("Train/relMAE", l1_loss_rel, on_step=True)

    def _compute_abs_rel_metric(
        self,
        metric_f: Callable,
        traj: torch.Tensor,
        duration: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor
    ) -> tuple[torch.Tensor]:
        # compute loss for each phase vector
        loss = metric_f(
            traj,
            pred * mask,
            reduction="none"
        )
        # get norm of traj. vectors residuals
        loss = loss.sum(dim=-1)
        # compute relative loss for traj. vectors
        # like ||resid|| / ||traj||
        rel_loss = metric_f(
            traj,
            torch.zeros_like(traj).to(traj),
            reduction="none"
        )
        rel_loss = rel_loss.sum(dim=-1) + 1e-6
        rel_loss = loss / rel_loss

        loss, rel_loss = list(
            [loss, rel_loss] |
            select(lambda l: l.sum(dim=-1) / duration) |
            select(lambda l: l.mean())
        )

        return loss, rel_loss

    def forward(self, z_0: torch.Tensor, num_steps: int) -> torch.Tensor:
        pred = odeint_adjoint(
                self.vf,
                z_0,
                torch.arange(num_steps).to(z_0),
                **self.odeint_kwargs
            )
        # make dims = (batch, duration, state_dim)
        pred = torch.swapdims(pred, 0, 1)

        return pred

    def configure_optimizers(self):
        return optim.AdamW(self.vf.parameters(), **self.optim_kwargs)
