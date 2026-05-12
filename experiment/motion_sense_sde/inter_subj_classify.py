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
from torch.utils.data import DataLoader, ConcatDataset
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
# from torchmetrics.regression import ...

from experiment.motion_sense_sde.utils.dataset import TrajectoryDataset
from experiment.motion_sense_sde.utils.field import FieldLitModule

import plotly.graph_objects as go
import rich
from rich.progress import track
import mlflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("act", type=str)
    args = parser.parse_args()
    config = OmegaConf.load("experiment/motion_sense_sde/config.yaml")

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment("motion_sense_sde")
    mlflow.start_run(
        run_name="inter_subj_classifiy",
        tags={"act": args.act}
    )

    act_fields_adapters = {}
    R_diags = {}
    test_trajectories = {}
    for subj_dir in Path(config.results_dir).iterdir():
        if not subj_dir.is_dir():
            continue
        subj = int(subj_dir.stem)
        act_dir = subj_dir / args.act

        test_dataset = TrajectoryDataset(
            config.data_dir, config.data_types, config.state_names, args.act,
            config.activity_codes[args.act][-1], subj
        )
        test_trajectories[subj] = test_dataset.traj

        field_path = act_dir / "best.ckpt"
        act_fields_adapters[subj] = FieldLitModule.load_from_checkpoint(
            field_path, weights_only=False,
            traj_mean=torch.zeros((test_dataset.d, ), dtype=torch.float32),
            traj_std=torch.zeros((test_dataset.d, ), dtype=torch.float32)
        ).to("cpu").eval()
        R_diags[subj] = pd.read_csv(
            act_dir / "R_diag.csv", index_col=0
        )["R_norm"].to_numpy()

    # perform classification
    cls_results = []
    for target_subj, test_traj in track(list(test_trajectories.items()), "Target subj"):
        test_traj_np = test_traj.numpy()
        t_mesh = np.arange(test_traj.shape[0]) * config.dt

        components_fig = []
        for i in range(test_traj_np.shape[1]):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t_mesh,
                y=test_traj_np[:, i],
                mode='lines',
                name=f"subj_{target_subj}_orig"
            ))
            components_fig.append(fig)

        for subj, field_adapter in track(list(act_fields_adapters.items()), "Pred subj", transient=True):
            mean = field_adapter.traj_mean.cpu()
            std = field_adapter.traj_std.cpu()
            test_traj_norm = ((test_traj - mean) / std).to(torch.float32).numpy()

            @torch.no_grad()
            def fx(x, dt_):
                xt = torch.from_numpy(x).to(torch.float32).unsqueeze(0)
                drift = field_adapter.field.f(torch.tensor(0.0), xt).squeeze(0).numpy()
                return x + drift * dt_
            
            points = MerweScaledSigmaPoints(field_adapter.d, alpha=.1, beta=2., kappa=-1)
            ukf = UnscentedKalmanFilter(
                field_adapter.d, field_adapter.d, field_adapter.dt,
                hx=identity, fx=fx, points=points
            )
            sigma = field_adapter.field.brownian_sigma.detach().numpy()
            ukf.Q = np.diag(sigma ** 2) * field_adapter.dt
            ukf.R = np.diag(R_diags[subj])
            ukf.x = test_traj_norm[0].copy()
            ukf.P = ukf.R.copy()

            mu, cov = ukf.batch_filter(test_traj_norm[1:])
            traj_smooth_norm, _, _ = ukf.rts_smoother(mu, cov)
            traj_smooth_norm = np.concat((test_traj_norm[0:1], traj_smooth_norm))

            traj_smooth = traj_smooth_norm * std.numpy() + mean.numpy()
            loss = float(np.sqrt(((traj_smooth - test_traj_np) ** 2).sum() / test_traj.shape[0]))
            cls_results.append((target_subj, subj, loss))

            for i, fig in enumerate(components_fig):
                fig.add_trace(go.Scatter(
                    x=t_mesh,
                    y=traj_smooth[:, i],
                    mode='lines',
                    name=f"subj_{subj}_smooth"
                ))

        for i, fig in enumerate(components_fig):
            state_name = field_adapter.state_names[i]
            fig.update_layout(
                xaxis_title="t", yaxis_title=state_name
            )
            mlflow.log_figure(fig, f"target_subj_{target_subj}/subj_{subj}_{state_name}.html")

    cls_results = pd.DataFrame(cls_results, columns=["subj_true", "subj", "loss"])
    argmin_indx = cls_results.groupby("subj_true")["loss"].idxmin()
    accuracy = (cls_results.loc[argmin_indx]["subj"] == cls_results.loc[argmin_indx]["subj_true"]).mean()
    mlflow.log_metric("accuracy", accuracy)

    save_dir = Path(config.results_dir)
    cls_results.to_csv(save_dir / f"{args.act}_inter.csv", index=False)
    mlflow.log_table(cls_results, "cls_results.json")

    print("Accuracy =", accuracy)
