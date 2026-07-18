import os
import argparse
from omegaconf import OmegaConf

import numpy as np
import pandas as pd

import torch
from torchdiffeq import odeint
from lightning.pytorch import seed_everything
from sklearn.metrics import accuracy_score
import statsmodels.api as sm

from rich.progress import track
import mlflow

from fields import matched_spectrum_trio

SEED = 847
NUM_SAMPLES = 500
d = 3
TEST_FRAC = 0.2
K = 5
BAND_FRAC = 0.1  # Sakoe-Chiba band radius as a fraction of the trajectory length


def dtw_one_to_many(query: np.ndarray, refs: np.ndarray, radius: int) -> np.ndarray:
    """DTW distance from one (T, d) query to every (S, d) reference in `refs` (N, S, d).

    The dynamic-programming recurrence is vectorised over the N references; a Sakoe-Chiba
    band of width `radius` restricts the warping path. Local cost is squared Euclidean and
    the returned distance is its accumulated square root.
    """
    N, S, _ = refs.shape
    T = query.shape[0]
    INF = 1e18
    D = np.full((N, T + 1, S + 1), INF, dtype=np.float64)
    D[:, 0, 0] = 0.0
    for i in range(1, T + 1):
        jmin = max(1, i - radius)
        jmax = min(S, i + radius)
        qi = query[i - 1]
        for j in range(jmin, jmax + 1):
            cost = np.sum((qi - refs[:, j - 1, :]) ** 2, axis=1)
            D[:, i, j] = cost + np.minimum(
                np.minimum(D[:, i - 1, j], D[:, i, j - 1]), D[:, i - 1, j - 1]
            )
    return np.sqrt(D[:, T, S])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sigma", type=float)
    args = parser.parse_args()
    config = OmegaConf.load("experiment/linear_systems/config.yaml")
    config.x0_sigma = 3.
    config.noise_sigma = args.sigma
    seed_everything(SEED)

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment("linear_systems")
    mlflow.start_run(run_name="knn_dtw_classify")
    mlflow.log_param("sigma", args.sigma)

    fields = matched_spectrum_trio(spacing=1.0)
    t_mesh = torch.arange(config.traj_len) * config.dt

    trajs, labels = [], []
    for label, field in enumerate(fields):
        x0 = torch.randn((NUM_SAMPLES, d)) * config.x0_sigma
        traj = odeint(field, x0, t_mesh)[1:]
        traj = traj + torch.randn_like(traj) * config.noise_sigma
        traj = traj.transpose(0, 1)
        trajs.append(traj)
        labels.append(torch.full((NUM_SAMPLES,), label, dtype=torch.long))
    trajs = torch.cat(trajs, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    n_total = len(trajs)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_total)
    n_test = int(n_total * TEST_FRAC)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    train_trajs, train_labels = trajs[train_idx], labels[train_idx]
    test_trajs, test_labels = trajs[test_idx], labels[test_idx]

    # per-coordinate standardisation with train statistics (so DTW is not scale-dominated)
    flat = train_trajs.reshape(-1, d)
    mean, std = flat.mean(axis=0), flat.std(axis=0).clip(min=1e-6)
    train_trajs = (train_trajs - mean) / std
    test_trajs = (test_trajs - mean) / std

    radius = max(1, int(config.traj_len * BAND_FRAC))
    y_pred = []
    for query in track(test_trajs, "k-NN + DTW"):
        dists = dtw_one_to_many(query, train_trajs, radius)
        nn_idx = np.argpartition(dists, K)[:K]
        votes = np.bincount(train_labels[nn_idx], minlength=len(fields))
        y_pred.append(votes.argmax())
    y_pred = np.array(y_pred)

    acc = accuracy_score(test_labels, y_pred)
    mlflow.log_param("k", K)
    mlflow.log_param("band_frac", BAND_FRAC)
    mlflow.log_metric("accuracy", acc)
    print("Test accuracy:", acc, "sigma", args.sigma)

    ci_low, ci_high = sm.stats.proportion_confint(
        (test_labels == y_pred).sum(), test_labels.shape[0], method="wilson"
    )
    df = pd.DataFrame({"sigma": [args.sigma], "accuracy": [acc], "ci_high": [ci_high], "ci_low": [ci_low]})
    df.to_csv(os.path.join(config.results_dir, "knn_dtw", f"sigma_{args.sigma:.5f}.csv"))

    mlflow.end_run()
