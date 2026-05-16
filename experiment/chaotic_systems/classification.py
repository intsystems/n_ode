import argparse
import os
from pathlib import Path
from omegaconf import OmegaConf
from itertools import chain
from toolz import pipe, identity
from toolz.curried import map as map_c

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchdiffeq import odeint
from lightning.pytorch import seed_everything
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from sklearn.metrics import accuracy_score

import plotly.graph_objects as go
from rich.progress import track

from fields import LorenzField, RosslerField, ChuaField

SEED = 847
NUM_SAMPLES = 1
TRAJ_LEN = 10
dt = 1e-1
d = 3
NOISE_SIGMA = 1e-1
X0_SIGMA = 1.

def np_field_adapter(torch_field: nn.Module):
    def field(x: np.ndarray, dt: float):
        x = torch.from_numpy(x)
        t_mesh = torch.tensor([0., dt])
        return odeint(torch_field, x, t_mesh)[-1].numpy()
    
    return field


if __name__ == "__main__":
    config = OmegaConf.load("experiment/chaotic_systems/config.yaml")
    seed_everything(SEED)

    y_true = []
    y_pred = []
    t_mesh = torch.arange(TRAJ_LEN) * dt
    for target_field_indx, target_field in enumerate(
        [LorenzField(), RosslerField(), ChuaField()]
    ):
        x0 = torch.randn((NUM_SAMPLES, d)) * X0_SIGMA
        target_trajs = odeint(target_field, x0, t_mesh)[1:]
        target_trajs += torch.randn_like(target_trajs) * NOISE_SIGMA
        target_trajs = target_trajs.transpose(0, 1)
        target_trajs_np = target_trajs.cpu().numpy()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=target_trajs_np[0, :, 0],
                y=target_trajs_np[0, :, 1],
                z=target_trajs_np[0, :, 2], mode='lines+markers',
                name=f"target_{target_field_indx}"
            )
        )

        pred_fields = [LorenzField(), RosslerField(), ChuaField()]
        true_pred_distances = []
        for pred_field_indx, pred_field in enumerate(pred_fields):
            points = MerweScaledSigmaPoints(d, alpha=.1, beta=2., kappa=-1)
            ukf = UnscentedKalmanFilter(
                d, d, dt,
                hx=identity, fx=np_field_adapter(pred_field), points=points
            )
            ukf.Q *= 1e-6
            ukf.R *= NOISE_SIGMA ** 2
            ukf.x = np.zeros((d, ))
            ukf.P *= X0_SIGMA ** 2

            distance = []
            for traj_num, traj in track(
                enumerate(target_trajs_np),
                f"Target: {target_field_indx}; Pred: {pred_field_indx}"
            ):
                mu, cov = ukf.batch_filter(traj)
                traj_smooth, _, _ = ukf.rts_smoother(mu, cov)
                distance.append(np.linalg.norm(traj_smooth - traj, 2))

                if traj_num == 0:
                    fig.add_trace(
                        go.Scatter3d(
                            x=traj_smooth[:, 0],
                            y=traj_smooth[:, 1],
                            z=traj_smooth[:, 2], mode='lines+markers',
                            name=f"pred_{pred_field_indx}"
                        )
                    )

            true_pred_distances.append(distance)
        true_pred_distances = np.array(true_pred_distances)

        pred_field_indx = np.argmin(true_pred_distances, axis=0)
        y_true.append(np.full((NUM_SAMPLES, ), target_field_indx))
        y_pred.append(pred_field_indx)

        fig.update_layout(
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z")
        )
        fig.update_traces(marker=dict(size=3))
        with open(
            os.path.join(config.results_dir, f"classifying_{target_field_indx}.html"),
            "w"
        ) as f:
            f.write(fig.to_html())

    y_true = np.concat(y_true)
    y_pred = np.concat(y_pred)
    print("Accuracy", accuracy_score(y_true, y_pred))