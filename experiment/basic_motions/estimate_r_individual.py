import argparse
import os
from pathlib import Path
from omegaconf import OmegaConf
from toolz import identity

import numpy as np
import pandas as pd

import torch
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from torchdiffeq import odeint

from experiment.basic_motions.utils.dataset import TrajectoryDataset
from experiment.basic_motions.utils.field import FieldLitModule, FieldAdapterForControl

import rich
console = rich.get_console()
from rich.progress import track
import mlflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label", type=str)
    args = parser.parse_args()
    config = OmegaConf.load("experiment/basic_motions/config.yaml")

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment("basic_motions")
    mlflow.start_run(
        run_name="estimate_r",
        tags={"label": args.label}
    )

    # R is measurement-noise covariance; with H = I and Q ~= 0 it equals the
    # covariance of the field's one-step prediction error. Estimate it on the
    # TRAIN split (where the field is well fit) using this class' own field.
    train_dataset = TrajectoryDataset(
        config.train_path, args.label,
        config.gyro_channels, config.accel_channels,
        dt=config.dt, window_size=1
    )
    d = train_dataset.d

    field_module = FieldLitModule.load_from_checkpoint(
        os.path.join(config.results_dir, args.label, "best.ckpt"), weights_only=False,
        traj_mean=torch.zeros((d,), dtype=torch.float32),
        traj_std=torch.zeros((d,), dtype=torch.float32),
        cont_control_mean=torch.zeros((train_dataset.num_cont_controls,), dtype=torch.float32),
        cont_control_std=torch.zeros((train_dataset.num_cont_controls,), dtype=torch.float32)
    ).to("cpu").eval()

    # accumulate the forward-filter innovations e_i = z_i - x_{i|i-1} over every
    # training trajectory; the innovation is exactly what R models (H = I)
    innovations = []
    for inst_idx in track(range(len(train_dataset.trajs)), "Train trajectory"):
        traj = train_dataset.trajs[inst_idx]
        spline_coefs = train_dataset.continuous_controls_spline_coeffs[inst_idx]

        cur_traj = (traj - field_module.traj_mean) / field_module.traj_std
        cur_traj = cur_traj.numpy()
        field_adapter_for_control = FieldAdapterForControl(
            field_module.field, spline_coefs,
            field_module.cont_control_mean, field_module.cont_control_std,
            config.dt
        )

        points = MerweScaledSigmaPoints(d, alpha=.1, beta=2., kappa=-1)
        ukf = UnscentedKalmanFilter(
            d, d, config.dt,
            hx=identity, fx=None, points=points
        )
        ukf.Q *= 1e-8
        # a small placeholder R only affects the update step; innovations are
        # read from the prior mean before update, so the estimate is R-free
        ukf.R = (torch.eye(d) * 1e-3).numpy().astype(np.float32)
        ukf.x = cur_traj[0].copy()
        ukf.P *= 1e-6

        @torch.no_grad()
        def fx_at(x, dt_, t):
            xt = torch.from_numpy(x).to(torch.float32)
            t_eval = torch.tensor([t, t + dt_], dtype=torch.float32)
            pred = odeint(
                field_adapter_for_control, xt, t_eval, method="rk4"
            )
            return pred[-1].numpy()

        for i in range(0, traj.shape[0] - 1):
            # drift used to propagate from step i to i + 1 is taken at time i * dt
            ukf.predict(fx=lambda x, dt_, t=i * config.dt: fx_at(x, dt_, t))
            # after predict, ukf.x is the prior mean x_{i+1|i}
            innovations.append(cur_traj[i + 1] - ukf.x)
            ukf.update(cur_traj[i + 1])

    innovations = np.stack(innovations)
    # diagonal R (per-channel); ddof=1 for an unbiased variance estimate
    R_diag = innovations.var(axis=0, ddof=1)

    R_df = pd.DataFrame({"R_diag": R_diag}, index=list(config.state_names))
    console.print(R_df)

    save_dir = Path(os.path.join(config.results_dir, args.label))
    save_dir.mkdir(parents=True, exist_ok=True)
    R_df.to_csv(save_dir / "R_diag.csv")

    mlflow.log_metric("num_innovations", innovations.shape[0])
    for name, val in zip(config.state_names, R_diag):
        mlflow.log_metric(f"R_diag/{name}", float(val))
    mlflow.log_table(R_df, "R_diag.json")
