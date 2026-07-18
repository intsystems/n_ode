import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from omegaconf import OmegaConf
from toolz import identity

import numpy as np
import pandas as pd

import torch
from torch import nn
from torchdiffeq import odeint
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

from experiment.ecg5000.utils.dataset import TakensTrajectoryDataset
from experiment.ecg5000.utils.field import FieldLitModule

from rich.progress import track

global ukf

def np_field_adapter(torch_field: nn.Module):
    def field(x: np.ndarray, dt: float):
        x = torch.from_numpy(x).unsqueeze(0).to(torch.float32)
        t_mesh = torch.tensor([0., dt])
        return odeint(
            torch_field, x, t_mesh, method="rk4"
        ).squeeze(1)[-1].numpy()
    
    return field

def worker_init(field_adapter: FieldLitModule, noise_sigma: np.ndarray):
    """ Makes UKF object in each worker
    """
    d = field_adapter.d
    dt = field_adapter.dt
    points = MerweScaledSigmaPoints(d, alpha=.1, beta=2., kappa=-1)
    global ukf
    ukf = UnscentedKalmanFilter(
        d, d, dt,
        hx=identity, fx=np_field_adapter(field_adapter.field), points=points
    )
    ukf.Q *= 1e-9
    ukf.R = np.diag(noise_sigma ** 2)
    ukf.P = np.diag(noise_sigma ** 2)

    ukf.traj_mean = field_adapter.traj_mean.numpy()
    ukf.traj_std = field_adapter.traj_std.numpy()

def compute_distance(traj: np.ndarray):
    global ukf
    traj_norm = (traj - ukf.traj_mean) / ukf.traj_std
    ukf.x = traj_norm[0]
    traj_norm = traj_norm[1:]
    
    mu, cov = ukf.batch_filter(traj_norm)
    traj_smooth, traj_covs, _ = ukf.rts_smoother(mu, cov)
    traj_smooth_unnorm = traj_smooth * ukf.traj_std + ukf.traj_mean
    traj_covs *= ukf.traj_std ** 2

    return np.linalg.norm(traj_smooth_unnorm - traj[1:], 1)
    # return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_label", type=int)
    parser.add_argument("pred_label", type=int)
    args = parser.parse_args()
    config = OmegaConf.load("experiment/ecg5000/config.yaml")

    test_dataset = TakensTrajectoryDataset(
        os.path.join(config.data_dir, "ECG5000_TEST.txt"),
        config.delay_dim, args.target_label
    )

    field_adapter = FieldLitModule.load_from_checkpoint(
        os.path.join(config.results_dir, str(args.pred_label), "best.ckpt"),
        weights_only=False,
        traj_mean=torch.zeros((config.delay_dim, ), dtype=torch.float32),
        traj_std=torch.zeros((config.delay_dim, ), dtype=torch.float32)
    ).to("cpu").eval()
    for param in field_adapter.parameters():
        param.requires_grad = False
    noise_sigma = np.ones((config.delay_dim,)).astype(np.float32) * 1e-3

    pool = ProcessPoolExecutor(
        max_workers=20, initializer=worker_init, initargs=(field_adapter, noise_sigma)
    )

    num_traj = len(test_dataset)
    results_futures = [
        pool.submit(compute_distance, test_dataset[i].numpy())
        for i in range(num_traj)
    ]
    results = []
    for f in track(results_futures, "Processing target trajectories"):
        results.append(f.result())
    results = pd.DataFrame(
        {"distance": results, "traj_num": list(range(num_traj))}
    )
    results["target"] = args.target_label
    results["pred"] = args.pred_label
    results.to_csv(
        os.path.join(config.results_dir, f"cls_{args.target_label}_{args.pred_label}.csv"),
        index=False
    )

    pool.shutdown()
