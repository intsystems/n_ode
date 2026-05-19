import numpy as np
from scipy import linalg
import torch
from torch.utils.data import Dataset


class TakensTrajectoryDataset(Dataset):
    def __init__(
        self, data_file: str, delay_dim: int,
        label: int, max_series: int = -1
    ):
        raw = np.loadtxt(data_file, dtype=np.float32)
        self.series = raw[raw[:, 0].astype(int) == label, 1:]
        self.series = self.series[:max_series]
        self.delay_dim = delay_dim

    def __getitem__(self, index):
        c = self.series[index, :self.delay_dim]
        r = self.series[index, self.delay_dim - 1:]
        return torch.from_numpy(
            linalg.hankel(c, r)
        ).transpose(0, 1)

    def __len__(self):
        return self.series.shape[0]
            

class TakensSlicedTrajectoryDataset(TakensTrajectoryDataset):
    def __init__(
        self, data_file: str, delay_dim: int,
        label: int, window_size: int, max_series: int = -1
    ):
        super().__init__(data_file, delay_dim, label, max_series)
        self.window_size = window_size
    
    def __getitem__(self, index):
        series_len = self.series.shape[1]
        takens_subtrajs_per_series = (series_len - self.delay_dim + 1) - \
            self.window_size + 1
        series_num, subtraj_start = divmod(index, takens_subtrajs_per_series)

        c = self.series[series_num, subtraj_start: subtraj_start + self.delay_dim]
        r = self.series[
            series_num,
            subtraj_start + self.delay_dim - 1: \
            subtraj_start + self.delay_dim - 1 + self.window_size
        ]
        return torch.from_numpy(
            linalg.hankel(c, r)
        ).transpose(0, 1)

    def __len__(self):
        series_len = self.series.shape[1]
        num_series = self.series.shape[0]
        return ((series_len - self.delay_dim + 1) - self.window_size + 1) * \
            num_series
