from typing import Callable
from pipe import select
from typing import Literal

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from torchdiffeq import odeint_adjoint

import lightning as L

from node.field_module import LitNode, get_trajectory_mask


class LitNodeSingleTraj(LitNode):
    def __init__(self, vf, optim_kwargs, odeint_kwargs, traj_num: int, val_traj_num: int):
        super().__init__(vf, optim_kwargs, odeint_kwargs)
        self.traj_num = traj_num
        self.val_traj_num = val_traj_num

    def on_train_start(self):
        if hasattr(self.logger, "log_text"):
            self.logger.log_text(f"traj_{self.traj_num}/Val_traj", columns=["val_traj_num"], data=[[self.val_traj_num]])

    def validation_step(self, *args, **kwargs):
        return self.training_step(*args, **kwargs)
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        traj, duration, subj, traj_num = batch
        self._log_metrics(traj, duration, "Train")
    
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        traj, duration, subj, traj_num = batch
        self._log_metrics(traj, duration, "Val")

    def _log_metrics(self, traj, duration, stage: Literal["Train", "Val"]):
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
        self.log(f"traj_{self.traj_num}/{stage}/MSE", l2_loss, prog_bar=True)
        self.log(f"traj_{self.traj_num}/{stage}/relMSE", l2_loss_rel)
        # compute l1-loss
        l1_loss, l1_loss_rel = self._compute_abs_rel_metric(
            F.l1_loss,
            traj,
            duration,
            pred,
            mask
        )
        self.log(f"traj_{self.traj_num}/{stage}/MAE", l1_loss)
        self.log(f"traj_{self.traj_num}/{stage}/relMAE", l1_loss_rel)
