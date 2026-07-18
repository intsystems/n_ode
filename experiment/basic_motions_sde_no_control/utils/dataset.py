import numpy as np
import torch
from torch.utils.data import Dataset


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

    The gyroscope channels form the SDE state. Unlike the controlled variant,
    the accelerometer is ignored: the field is autonomous (``dx = f(x) dt +
    g dW``), so there is no control and no continuous-control spline. Windows
    never cross instance boundaries.
    """

    def __init__(
        self, ts_path: str, label: str,
        gyro_channels: list[int],
        dt: float = 0.1, window_size: int = 16
    ):
        instances, labels = load_ts(ts_path)

        self.trajs = []                  # gyroscope state, list of (T, d)
        for inst, inst_label in zip(instances, labels):
            if inst_label != label:
                continue
            traj = torch.from_numpy(inst[:, gyro_channels]).to(torch.float32)
            self.trajs.append(traj)

        self.dt = dt
        self.window_size = window_size
        self.d = self.trajs[0].shape[1]

        # flat (instance, window_start) index over every valid window
        self._windows = [
            (i, start)
            for i, traj in enumerate(self.trajs)
            for start in range(traj.shape[0] - window_size + 1)
        ]

    def __getitem__(self, index):
        i, start = self._windows[index]
        return self.trajs[i][start : start + self.window_size]

    def __len__(self):
        return len(self._windows)
