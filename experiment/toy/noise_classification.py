import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy import linalg

from rich.progress import track

from filterpy.kalman import KalmanFilter

SEED = 132
np.random.seed(123)

lambd = -18.
X0_SIGMA = 1.
X0_MEAN = 1.
T = 1
N = 11
SIGMA = 1.
t = np.linspace(0, T, N)

def f(t, x):
    return lambd * x

x_euler = X0_MEAN * ((1 + lambd * (T / (N - 1))) ** np.arange(N))

def study():
    x0 = X0_MEAN + X0_SIGMA * np.random.randn()
    x_true = x0 * np.exp(lambd * t)
    
    x_obs = x_true + SIGMA * np.random.randn(*x_true.shape)
    x_obs[0] = X0_MEAN
    
    euler_by_step = [X0_MEAN]
    for x0_cur in x_obs[:-1]:
        euler_by_step.append(
            x0_cur * (1 + lambd * (T / (N - 1)))
        )
    euler_by_step = np.array(euler_by_step)

    smoother = KalmanFilter(dim_x=1, dim_z=1)
    smoother.x = np.array([X0_MEAN])
    smoother.P *= X0_SIGMA ** 2
    smoother.F = np.array([[1. + lambd * (T / (N - 1))]])
    smoother.H = np.array([[1.]])
    smoother.Q *= 0.
    smoother.R *= SIGMA ** 2
    (mu, cov, _, _) = smoother.batch_filter(x_obs[1:])
    x_smoothed, _, _, _ = smoother.rts_smoother(mu, cov)
    x_smoothed = x_smoothed.flatten()
    x_smoothed = np.concat((np.array([X0_MEAN]), x_smoothed))

    return np.argmin(np.array([
        linalg.norm(x_true - x_euler), 
        linalg.norm(x_true - euler_by_step), 
        linalg.norm(x_true - x_smoothed)
    ]))


if __name__ == "__main__":
    NUM_STUDIES = 10000
    results = np.array([study() for _ in range(NUM_STUDIES)])
    method_indxs, method_count = np.unique(results, return_counts=True)
    method_count = method_count.astype(float)
    method_count /= NUM_STUDIES
    method_names = ["euler", "euler_step", "smooth"]
    for method_indx, method_count in zip(method_indxs, method_count):
        print(method_names[method_indx], method_count)

