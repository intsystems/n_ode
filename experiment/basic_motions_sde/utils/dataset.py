import numpy as np
import torch
from torch.utils.data import Dataset
from torchcubicspline import natural_cubic_spline_coeffs


def load_ts(path: str) -> tuple[list[np.ndarray], list[str]]:
    """Parse a sktime/aeon ``.ts`` file into per-instance arrays and labels.

    Each ``@data`` line stores one multivariate instance as
    ``dim_1:dim_2:...:dim_D:label`` where every dimension is a comma separated
    series. Returns a list of ``(T, num_channels)`` arrays and the matching
    string labels.
    """
    instances, labels = [], []
    with open(path) as f:
        in_data = False
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@"):
                if line.lower() == "@data":
                    in_data = True
                continue
            if not in_data:
                continue
            *dim_strs, label = line.split(":")
            channels = np.array(
                [[float(v) for v in dim.split(",")] for dim in dim_strs],
                dtype=np.float32,
            )  # (num_channels, T)
            instances.append(channels.T)  # (T, num_channels)
            labels.append(label)
    return instances, labels


class TrajectoryDataset(Dataset):
    """All trajectories of a single BasicMotions class from one split.

    The gyroscope channels form the SDE state, the accelerometer channels form
    the (continuous) control. BasicMotions has no per-subject metadata, so there
    are no discrete controls. Windows never cross instance boundaries.
    """

    def __init__(
        self, ts_path: str, label: str,
        gyro_channels: list[int], accel_channels: list[int],
        dt: float = 0.1, window_size: int = 16
    ):
        instances, labels = load_ts(ts_path)

        self.trajs = []                  # gyroscope state, list of (T, d)
        self.controls = []               # acceleration control, list of (T, num_cont)
        self.continuous_controls_spline_coeffs = []  # per-instance list of coeff tensors
        for inst, inst_label in zip(instances, labels):
            if inst_label != label:
                continue
            traj = torch.from_numpy(inst[:, gyro_channels]).to(torch.float32)
            control = torch.from_numpy(inst[:, accel_channels]).to(torch.float32)
            # discard t-values, get only spline coeffs
            coeffs = natural_cubic_spline_coeffs(
                torch.arange(control.shape[0]) * dt, control
            )[1:]
            coeffs = [c.to(torch.float32) for c in coeffs]

            self.trajs.append(traj)
            self.controls.append(control)
            self.continuous_controls_spline_coeffs.append(coeffs)

        self.dt = dt
        self.window_size = window_size
        self.d = self.trajs[0].shape[1]
        self.num_cont_controls = self.controls[0].shape[1]
        self.num_discr_controls = 0

        # flat (instance, window_start) index over every valid window
        self._windows = [
            (i, start)
            for i, traj in enumerate(self.trajs)
            for start in range(traj.shape[0] - window_size + 1)
        ]

    def __getitem__(self, index):
        i, start = self._windows[index]
        return self.trajs[i][start : start + self.window_size], \
            *[
                coef[start : start + self.window_size - 1]      # splines are subtraj_len - 1
                for coef in self.continuous_controls_spline_coeffs[i]
            ]

    def __len__(self):
        return len(self._windows)
