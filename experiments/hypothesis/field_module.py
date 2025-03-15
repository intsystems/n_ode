from typing import Callable
from pipe import select

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from torchdiffeq import odeint_adjoint

import lightning as L

from node.field_module import LitNode


class LitNodeHype(LitNode):
    """ Complement base class with validation which computes liklyhood for full trajectories.
    """
    def validation_step(self, batch, batch_idx):
        traj, duration, subj, traj_num = batch
        z_0 = traj[:, 0, :]

        pred = self._odeint(z_0)
        mask = self._get_trajectory_mask(duration, traj)

        # save tensors to compute liklyhood for the whole traj
        self._cur_traj_info.append([traj_num, traj, duration, pred])

        # compute l2-loss for epoch
        l2_loss, l2_loss_rel = self._compute_abs_rel_metric(
            F.mse_loss,
            traj,
            duration,
            pred,
            mask
        )
        self.log("Val/MSE", l2_loss, on_epoch=True)
        self.log("Val/relMSE", l2_loss_rel, on_epoch=True)
        # compute l1-loss for epoch
        l1_loss, l1_loss_rel = self._compute_abs_rel_metric(
            F.l1_loss,
            traj,
            duration,
            pred,
            mask
        )
        self.log("Val/MAE", l1_loss, on_epoch=True)
        self.log("Val/relMAE", l1_loss_rel, on_epoch=True)

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
        traj, duration, subj, traj_num = batch

        if not hasattr(self, "_cur_traj_num"):
            self._cur_traj_num = traj_num[0]
            self._cur_traj_info = []

        if not torch.all(traj_num == self._cur_traj_num):
            # last batch contains part of other trajectory
            # we now save other trajectory part
            # and compute liklyhood of full current trajctory
            last_traj_info = self._cur_traj_info[-1]
            self._cur_traj_info.pop()
            split_indx = (traj_num == self._cur_traj_num).argmax()
            self._cur_traj_info.append(list(
                last_traj_info | select(lambda x: x[:split_indx])
            ))

            lh = self._compute_lh()
            self.log("Val/traj_liklyhood", lh, on_epoch=True)

            # save other trajectory part and num
            self._cur_traj_info = list(
                last_traj_info | select(lambda x: x[split_indx:])
            )
            self._cur_traj_num = traj_num[-1]

    def on_validation_epoch_end(self):
        # compute lh for last trajectory
        lh = self._compute_lh()
        self.log("Val/traj_liklyhood", lh, on_epoch=True)

        # clean up after validation epoch
        del self._cur_traj_info
        del self._cur_traj_num
        
    def _compute_lh(self) -> torch.Tensor:
        traj_num, traj, duration, pred = list(
            zip(*self._cur_traj_info) |
            select(lambda x: torch.concat(x))
        )
        mask = self._get_trajectory_mask(duration, traj)

        lh = F.mse_loss(
            traj,
            pred * mask,
            reduction="sum"
        )
        
        return lh
    
