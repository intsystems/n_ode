import os
import argparse
from omegaconf import OmegaConf
from toolz import identity
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchdiffeq import odeint
from lightning.pytorch import seed_everything
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from sklearn.metrics import accuracy_score
import statsmodels.api as sm

from rich.progress import track
import mlflow

from fields import matched_spectrum_trio, simulate_sde_paths

SEED = 847
WOKRERS = 5
NUM_SAMPLES = 100
d = 3

def np_field_adapter(torch_field: nn.Module):
    def field(x: np.ndarray, dt: float):
        x = torch.from_numpy(x).float()
        t_mesh = torch.tensor([0., dt])
        return odeint(torch_field, x, t_mesh)[-1].numpy()

    return field

def compute_distance(pred_field, traj, dt, sigma_proc, obs_sigma, x0_sigma):
    points = MerweScaledSigmaPoints(d, alpha=.1, beta=2., kappa=-1)
    ukf = UnscentedKalmanFilter(
        d, d, dt,
        hx=identity, fx=np_field_adapter(pred_field), points=points
    )
    # process noise from the SDE diffusion (Q = sigma_proc^2 dt I; constant -> stationary smoother)
    ukf.Q = np.eye(d) * sigma_proc ** 2 * dt
    ukf.R *= obs_sigma ** 2
    ukf.x = np.zeros((d, ))
    ukf.P *= x0_sigma ** 2

    mu, cov = ukf.batch_filter(traj)
    traj_smooth, _, _ = ukf.rts_smoother(mu, cov)
    return np.linalg.norm(traj_smooth - traj, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sigma", type=float)
    args = parser.parse_args()
    config = OmegaConf.load("experiment/linear_systems_sde/config.yaml")
    config.x0_sigma = 3.
    config.obs_sigma = args.sigma  # swept observation noise; sigma_proc stays fixed from config
    seed_everything(SEED)
    pool = ProcessPoolExecutor(WOKRERS)

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment("linear_systems_sde")
    mlflow.start_run(
        run_name="kalman_classifiy",
    )
    mlflow.log_param("sigma", args.sigma)  # observation noise (swept)
    mlflow.log_param("sigma_proc", config.sigma_proc)  # diffusion noise (fixed)

    y_true = []
    y_pred = []
    t_mesh = torch.arange(config.traj_len) * config.dt
    candidate_fields = matched_spectrum_trio(spacing=1.0)
    for target_field_indx, target_field in enumerate(candidate_fields):
        x0 = torch.randn((NUM_SAMPLES, d)) * config.x0_sigma
        target_trajs = simulate_sde_paths(target_field, x0, t_mesh, config.sigma_proc)[1:]
        target_trajs += torch.randn_like(target_trajs) * config.obs_sigma
        target_trajs = target_trajs.transpose(0, 1)
        target_trajs_np = target_trajs.cpu().numpy()

        pred_fields = matched_spectrum_trio(spacing=1.0)
        true_pred_distances = []
        for pred_field_indx, pred_field in enumerate(pred_fields):
            distance_tasks = [
                pool.submit(
                    compute_distance, pred_field, traj, config.dt,
                    config.sigma_proc, config.obs_sigma, config.x0_sigma
                )
                for traj in target_trajs_np
            ]
            distance = []
            for f in track(distance_tasks, f"Target: {target_field_indx}; Pred: {pred_field_indx}"):
                distance.append(f.result())
            true_pred_distances.append(distance)
        true_pred_distances = np.array(true_pred_distances)

        pred_field_indx = np.argmin(true_pred_distances, axis=0)
        y_true.append(np.full((NUM_SAMPLES, ), target_field_indx))
        y_pred.append(pred_field_indx)

    y_true = np.concat(y_true)
    y_pred = np.concat(y_pred)
    acc = accuracy_score(y_true, y_pred)
    mlflow.log_metric("accuracy", acc)
    print("Accuracy", acc, "simga", args.sigma)

    ci_low, ci_high = sm.stats.proportion_confint(
        (y_true == y_pred).sum(), y_true.shape[0], method="wilson"
    )
    df = pd.DataFrame({"sigma": [args.sigma], "accuracy": [acc], "ci_high": [ci_high], "ci_low": [ci_low]})
    df.to_csv(os.path.join(config.results_dir, "kalman", f"sigma_{args.sigma:.5f}.csv"))

    pool.shutdown()
