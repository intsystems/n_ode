import numpy as np
import torch
import torch.nn as nn

from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat
import rich
console = rich.get_console()


class SmoothedTrajectoryLoss(Metric):
    def __init__(self, kalman, max_len = None):
        super().__init__()
        # a.k.a. from filterpy.kalman import KalmanFilter
        self.kalman = kalman
        self.add_state("traj", default=[], dist_reduce_fx="cat")
        self.max_len = max_len
    
    def update(self, traj_batch: torch.Tensor):
        self.traj.append(traj_batch)
    
    def compute(self):
        traj = dim_zero_cat(self.traj).numpy()
        if self.max_len is not None:
            traj = traj[:self.max_len]
        self.kalman.x = traj[0]
        self.kalman.P = self.kalman.Q.copy()
        with console.status("Running smoother forward"):
            mu, cov = self.kalman.batch_filter(traj[1:])
        with console.status("Running smoother backward"):
            traj_smooth, _, _ = self.kalman.rts_smoother(mu, cov)
        traj_smooth = np.concat((traj[0:1], traj_smooth))

        # rmse
        loss = nn.functional.mse_loss(
            torch.from_numpy(traj_smooth), torch.from_numpy(traj),
        ).sqrt()
        return loss, traj, traj_smooth
