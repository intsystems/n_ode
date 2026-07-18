import argparse
import os
from pathlib import Path
from omegaconf import OmegaConf
from toolz import identity

import numpy as np
import pandas as pd

import torch
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints, unscented_transform
from torchdiffeq import odeint

from experiment.basic_motions_sde.utils.dataset import TrajectoryDataset
from experiment.basic_motions_sde.utils.field import FieldLitModule, FieldAdapterForControl

import plotly.graph_objects as go
import rich
console = rich.get_console()
from rich.progress import track
import mlflow


def rts_smoother_nonstationary(ukf, Xs, Ps, fx_at, dt, Qs=None):
    """RTS smoother for a UKF with a time-varying (nonstationary) transition.

    Mirrors ``filterpy``'s ``UnscentedKalmanFilter.rts_smoother`` but, instead of
    reusing a single ``ukf.fx``, evaluates ``fx_at(x, dt, t)`` at the time
    ``t = k * dt`` that was used to propagate the state from step ``k`` to
    ``k + 1`` during the forward filtering pass.
    """
    n, dim_x = Xs.shape
    if Qs is None:
        Qs = [ukf.Q] * n

    Ks = np.zeros((n, dim_x, dim_x))
    num_sigmas = ukf._num_sigmas
    xs, ps = Xs.copy(), Ps.copy()
    sigmas_f = np.zeros((num_sigmas, dim_x))

    for k in reversed(range(n - 1)):
        # create sigma points from the filtered estimate, pass them through the
        # transition evaluated at the time used to go from step k to k + 1
        sigmas = ukf.points_fn.sigma_points(xs[k], ps[k])
        for i in range(num_sigmas):
            sigmas_f[i] = fx_at(sigmas[i], dt, k * dt)

        xb, Pb = unscented_transform(
            sigmas_f, ukf.Wm, ukf.Wc, Qs[k], ukf.x_mean, ukf.residual_x
        )

        # cross covariance between filtered state and predicted next state
        Pxb = 0
        for i in range(num_sigmas):
            y = ukf.residual_x(sigmas_f[i], xb)
            z = ukf.residual_x(sigmas[i], Xs[k])
            Pxb += ukf.Wc[i] * np.outer(z, y)

        # smoother gain and smoothed estimate
        K = Pxb @ ukf.inv(Pb)
        xs[k] += K @ ukf.residual_x(xs[k + 1], xb)
        ps[k] += K @ (ps[k + 1] - Pb) @ K.T
        Ks[k] = K

    return xs, ps, Ks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label", type=str)
    args = parser.parse_args()
    config = OmegaConf.load("experiment/basic_motions_sde/config.yaml")

    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment("basic_motions_sde")
    mlflow.start_run(
        run_name="test_subj_classify",
        tags={"label": args.label}
    )

    # every test instance whose true class is `label`
    test_dataset = TrajectoryDataset(
        config.test_path, args.label,
        config.gyro_channels, config.accel_channels,
        dt=config.dt, window_size=1
    )
    d = test_dataset.d
    num_cont_controls = test_dataset.num_cont_controls

    # one field per candidate class
    label_fields = {}
    for label in config.classes:
        field_module = FieldLitModule.load_from_checkpoint(
            os.path.join(config.results_dir, label, "best.ckpt"), weights_only=False,
            traj_mean=torch.zeros((d,), dtype=torch.float32),
            traj_std=torch.zeros((d,), dtype=torch.float32),
            cont_control_mean=torch.zeros((num_cont_controls,), dtype=torch.float32),
            cont_control_std=torch.zeros((num_cont_controls,), dtype=torch.float32)
        ).to("cpu").eval()
        label_fields[label] = field_module

    cls_results = []
    for inst_idx in track(range(len(test_dataset.trajs)), "Test trajectory"):
        traj = test_dataset.trajs[inst_idx]
        spline_coefs = test_dataset.continuous_controls_spline_coeffs[inst_idx]
        t_mesh = np.arange(traj.shape[0]) * config.dt

        components_fig = []
        for i in range(traj.shape[1]):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t_mesh,
                y=traj.numpy()[:, i],
                mode='lines',
                name="orig"
            ))
            components_fig.append(fig)

        for label, field_module in list(label_fields.items()):
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
            ukf.Q = np.eye(d)
            ukf.R *= 1e-3
            ukf.x = cur_traj[0].copy()
            ukf.P *= 1e-3

            @torch.no_grad()
            def fx_at(x, dt_, t):
                xt = torch.from_numpy(x).to(torch.float32)
                t_eval = torch.tensor([t, t + dt_], dtype=torch.float32)
                pred = odeint(
                    field_adapter_for_control.f, xt, t_eval, rtol=1e-6, atol=1e-6,
                    method="euler"
                )
                return pred[-1].numpy()

            # store the initial state so the smoother also covers t = 0
            mu = [cur_traj[0].copy()]
            cov = [ukf.P.copy()]
            Qs = []
            for i in range(0, traj.shape[0] - 1):
                # diffusion-driven process noise at time i * dt
                ukf.Q = np.eye(d) * config.dt * field_adapter_for_control.g(
                    torch.tensor(i * config.dt, dtype=torch.float32),
                    # dummy x: diffusion depends only on the control
                    torch.zeros((d,), dtype=torch.float32)
                ).detach().numpy() ** 2
                # drift used to propagate from step i to i + 1 is taken at time i * dt
                ukf.predict(fx=lambda x, dt_, t=i * config.dt: fx_at(x, dt_, t))
                ukf.update(cur_traj[i + 1])
                mu.append(ukf.x.copy())
                cov.append(ukf.P.copy())
                Qs.append(ukf.Q.copy())

            mu = np.stack(mu)
            cov = np.stack(cov)
            traj_smooth_norm, traj_covs, _ = rts_smoother_nonstationary(
                ukf, mu, cov, fx_at, config.dt, Qs
            )

            traj_smooth_unnorm = traj_smooth_norm * field_module.traj_std.numpy() + \
                field_module.traj_mean.numpy()
            for i, fig in enumerate(components_fig):
                comp_sigma = np.sqrt(np.array([cov[i, i] for cov in traj_covs]))
                fig.add_trace(go.Scatter(
                    x=t_mesh,
                    y=traj_smooth_unnorm[:, i],
                    error_y=dict(
                        type='data',
                        array=comp_sigma / 2,
                        arrayminus=comp_sigma / 2,
                        visible=True
                    ),
                    mode='lines',
                    name=label
                ))
            
            loss = float(np.abs(traj_smooth_unnorm - traj.numpy()).sum() / traj_smooth_unnorm.shape[0])
            cls_results.append((inst_idx, label, loss))

        for i, fig in enumerate(components_fig):
            state_name = config.state_names[i]
            fig.update_layout(
                xaxis_title="t", yaxis_title=state_name
            )
            mlflow.log_figure(fig, f"{args.label}_inst_{inst_idx}/{state_name}.html")

    cls_results = pd.DataFrame(cls_results, columns=["inst", "pred_label", "loss"])
    cls_results["true_label"] = args.label
    argmin_indx = cls_results.groupby(["inst"])["loss"].idxmin()
    accuracy = (cls_results.loc[argmin_indx]["pred_label"] == cls_results.loc[argmin_indx]["true_label"]).mean()
    mlflow.log_metric("accuracy", accuracy)

    save_dir = Path(config.results_dir)
    cls_results.to_csv(save_dir / f"{args.label}_test.csv", index=False)
    mlflow.log_table(cls_results, "cls_results.json")

    print("Accuracy =", accuracy)
