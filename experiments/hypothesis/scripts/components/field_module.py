from typing import Callable
from warnings import deprecated
from pipe import select

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from torchdiffeq import odeint_adjoint

import lightning as L

from node.field_module import LitNode, get_trajectory_mask

@deprecated("Useless function")
def compute_lh(
    traj: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    return F.mse_loss(
        traj,
        pred * mask,
        reduction="sum"
    )


class LitNodeHype(LitNode):
    """ Complement base class with validation which computes liklyhood for full trajectories.
    """
    def validation_step(self, batch, batch_idx):
        traj, duration, subj, traj_num = batch
        z_0 = traj[:, 0, :]

        pred = self.forward(z_0, num_steps=traj.shape[1])
        mask = get_trajectory_mask(duration, traj)

        for loss_name, loss_f in self.loss_funcs.items():
            loss_abs, loss_rel = self._compute_abs_rel_metric(
                loss_f,
                traj,
                duration,
                pred,
                mask
            )

            self.log(f"Val/{loss_name}", loss_abs, on_epoch=True)
            self.log(f"Val/rel{loss_name}", loss_rel, on_epoch=True)
