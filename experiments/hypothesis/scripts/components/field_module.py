from typing import Callable
from pipe import select

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from torchdiffeq import odeint_adjoint

import lightning as L

from node.field_module import LitNode, get_trajectory_mask

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
    def __init__(self, vf, loss_funcs, optim_kwargs, odeint_kwargs):
        super().__init__(vf, loss_funcs, optim_kwargs, odeint_kwargs)

        # counter and container for computing full trajectories liklyhood
        self._cur_traj_num: int = None
        self._cur_traj_info: list[list[torch.Tensor]] = []

    def validation_step(self, batch, batch_idx):
        traj, duration, subj, traj_num = batch
        z_0 = traj[:, 0, :]

        pred = self.forward(z_0, num_steps=traj.shape[1])
        mask = get_trajectory_mask(duration, traj)

        # save tensors to compute liklyhood for the whole traj
        self._cur_traj_info.append([traj_num, traj, duration, pred])

        for loss_name, loss_f in self.loss_funcs.items():
            loss_abs, loss_rel = self._compute_abs_rel_metric(
                loss_f,
                traj,
                duration,
                pred,
                mask
            )

            self.log(f"Train/{loss_name}", loss_abs, on_step=True)
            self.log(f"Train/rel{loss_name}", loss_rel, on_step=True)

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
        traj, duration, subj, traj_num = batch

        if self._cur_traj_num is None:
            self._cur_traj_num = traj_num[0]

        if not torch.all(traj_num == self._cur_traj_num):
            # last batch contains part of other trajectory
            # we now save other trajectory part
            # and compute liklyhood of full current trajctory
            last_traj_info: list[list[torch.Tensor]] = self._cur_traj_info[-1]
            self._cur_traj_info.pop()
            split_indx = (traj_num == self._cur_traj_num).to(torch.int32).argmax() + 1
            self._cur_traj_info.append(list(
                last_traj_info | select(lambda x: x[:split_indx])
            ))

            _, full_traj, full_duration, full_pred = list(
                zip(*self._cur_traj_info) |
                select(lambda x: torch.concat(x))
            )
            full_mask = get_trajectory_mask(full_duration, full_traj)

            optim_loss = list(self.loss_funcs.values())[0]
            lh = optim_loss(
                full_traj,
                full_pred * full_mask,
                reduction="sum"
            )
            self.log("Val/traj_liklyhood", lh, on_epoch=True)

            # save other trajectory part and num
            self._cur_traj_info = [list(
                last_traj_info | select(lambda x: x[split_indx:])
            )]
            self._cur_traj_num = traj_num[-1]

    def on_validation_epoch_end(self):
        # compute lh for last trajectory
        _, full_traj, full_duration, full_pred = list(
            zip(*self._cur_traj_info) |
            select(lambda x: torch.concat(x))
        )
        full_mask = get_trajectory_mask(full_duration, full_traj)
        optim_loss = list(self.loss_funcs.values())[0]
        lh = optim_loss(
                full_traj,
                full_pred * full_mask,
                reduction="sum"
            )
        self.log("Val/traj_liklyhood", lh, on_epoch=True)

        # clean up after validation epoch
        self._cur_traj_info.clear()
        self._cur_traj_num = None
