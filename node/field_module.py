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
        loss_funcs: dict[str, Callable],
        optim_kwargs: dict,
        odeint_kwargs: dict
    ):
        super().__init__()

        self.vf = vf
        self.loss_funcs = loss_funcs
        self.optim_kwargs = optim_kwargs
        self.odeint_kwargs = odeint_kwargs

        self.save_hyperparameters(ignore=["vf", "loss_funcs"], logger=False)

    def training_step(self, batch, batch_idx):
        traj, duration, subj, traj_num = batch
        z_0 = traj[:, 0, :]

        pred = self(z_0, num_steps=traj.shape[1])
        mask = get_trajectory_mask(duration, traj)
        
        optim_loss = None
        # dicts are ordered by insertion order in Python 3.7+
        for i, (loss_name, loss_f) in enumerate(self.loss_funcs.items()):
            loss_abs, loss_rel = self._compute_abs_rel_metric(
                loss_f,
                traj,
                duration,
                pred,
                mask
            )

            if i == 0:
                optim_loss = loss_abs
                prog_bar = True
            else:
                prog_bar = False
            self.log(f"Train/{loss_name}", loss_abs, prog_bar=prog_bar, on_step=True)
            self.log(f"Train/rel{loss_name}", loss_rel, on_step=True)

        return optim_loss

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
        traj_norm = metric_f(
            traj,
            torch.zeros_like(traj).to(traj),
            reduction="none"
        )
        EPS = 1e-6
        traj_norm = traj_norm.sum(dim=-1) + EPS
        rel_loss = loss / traj_norm

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
                options={"dtype": torch.float32},
                **self.odeint_kwargs
            )
        # make dims = (batch, duration, state_dim)
        pred = torch.swapdims(pred, 0, 1)

        return pred

    def configure_optimizers(self):
        return optim.AdamW(self.vf.parameters(), **self.optim_kwargs)
