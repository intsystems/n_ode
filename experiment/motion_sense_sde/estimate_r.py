import argparse
import os
from pathlib import Path
from omegaconf import OmegaConf
from toolz import identity

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

from experiment.motion_sense_sde.utils.dataset import TrajectoryDataset
from experiment.motion_sense_sde.utils.field import FieldLitModule

import rich
console = rich.get_console()
import mlflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("act", type=str)
    parser.add_argument("subj", type=int)
    args = parser.parse_args()
    config = OmegaConf.load("experiment/motion_sense_sde/config.yaml")

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment("motion_sense_sde")
    mlflow.start_run(
        run_name="R_learn",
        tags={"subj": args.subj, "act": args.act}
    )

    test_dataset = TrajectoryDataset(
        config.data_dir, config.data_types, config.state_names,
        args.act, config.activity_codes[args.act][-1], args.subj
    )
    d = test_dataset.d

    ckpt = os.path.join(config.results_dir, str(args.subj), args.act, "best.ckpt")
    field_mod = FieldLitModule.load_from_checkpoint(
        ckpt, weights_only=False,
        traj_mean=torch.zeros((d,), dtype=torch.float32),
        traj_std=torch.zeros((d,), dtype=torch.float32),
    ).eval()
    dt = field_mod.dt
    std = field_mod.traj_std.numpy()

    # observations in the normalized state space the field was trained in
    TRAJ_TRUNCATION = 100
    y = ((test_dataset.traj - field_mod.traj_mean) / field_mod.traj_std).numpy()
    y = y[:TRAJ_TRUNCATION]

    # Q from the learned diffusion; H = I (state == observation)
    sigma = field_mod.field.brownian_sigma.detach().numpy()
    Q = np.diag(sigma ** 2) * dt

    @torch.no_grad()
    def fx(x, dt_):
        xt = torch.from_numpy(x).to(torch.float32).unsqueeze(0)
        drift = field_mod.field.f(torch.tensor(0.0), xt).squeeze(0).numpy()
        return x + drift * dt_

    def build_ukf(R):
        points = MerweScaledSigmaPoints(d, alpha=.1, beta=2., kappa=-1)
        ukf = UnscentedKalmanFilter(d, d, dt, hx=identity, fx=fx, points=points)
        ukf.Q = Q
        ukf.R = R
        ukf.x = y[0].copy()
        ukf.P = Q.copy()
        return ukf

    def neg_loglik(log_r: np.ndarray) -> float:
        R = np.diag(np.exp(log_r))
        ukf = build_ukf(R)
        # negative log likelyhood
        nll = 0.0
        const = d * np.log(2.0 * np.pi)
        for z in y[1:]:
            ukf.predict()
            S = ukf.P + R                       # H = I  =>  S = P_{t|t-1} + R
            e = z - ukf.x                       # innovation
            try:
                c, low = cho_factor(S)
            except np.linalg.LinAlgError:
                return np.inf
            logdet = 2.0 * np.log(np.diag(c)).sum()
            quad = e @ cho_solve((c, low), e)
            nll += 0.5 * (const + logdet + quad)
            ukf.update(z)
        return float(nll)

    log_r0 = np.log(np.full(d, 1e-2))
    with console.status("Learning R"):
        res = minimize(neg_loglik, log_r0, method="L-BFGS-B")
    R_diag_norm = np.exp(res.x)
    R_diag_raw = R_diag_norm * std ** 2

    mlflow.log_metric("nll", res.fun)
    mlflow.log_metric("converged", res.success)
    print(f"converged: {res.success}, nll: {res.fun:.4f}")
    print("R diagonal (normalized):", R_diag_norm)
    print("R diagonal (raw units):", R_diag_raw)

    save_dir = Path(os.path.join(config.results_dir, str(args.subj), args.act))
    save_dir.mkdir(parents=True, exist_ok=True)
    R_df = pd.DataFrame(
        {"R_norm": R_diag_norm, "R_raw": R_diag_raw},
        index=config.state_names
    )
    R_df.to_csv(save_dir / "R_diag.csv")
    mlflow.log_table(R_df, "R_diag.json")
