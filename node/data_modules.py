from typing import Optional
from pathlib import Path
from shutil import rmtree
from copy import deepcopy
from omegaconf import OmegaConf

from pipe import select, izip
from itertools import starmap

import numpy as np
from scipy import linalg

import torch
from torch.utils.data import Dataset, random_split, DataLoader

import lightning as L

from .raw_data_loading import creat_time_series, set_data_types
from .traj_build import takens_traj

import warnings

class ActivityTrajDataset(Dataset):
    def __init__(
        self,
        act: str,
        dim: int,
        max_len: int,
        data_path: Path,
        save_dir: Path,            # storage file
        mmap: Optional[bool],
        data_type: str = "rotationRate"
    ):
        super().__init__()

        # save info about dataset + engineering staff
        self.act = act
        self.dim = dim
        self.max_len = max_len
        self.data_type = data_type
        self.save_dir = save_dir
        self._mmap = mmap
        # load config files for data
        self.data_params = OmegaConf.load(data_path / "dataset_params.yaml")

        self._data_files = list(
            ("trajs.pt", "durations.pt", "subjs.pt") |
            select(lambda data_name: Path(save_dir / data_name))
        )

        # build trajectories if it's not exist or empty
        if not self.save_dir.exists() or len(list(self.save_dir.iterdir())) == 0:
            print("Building trajectories")

            self.save_dir.mkdir(parents=True, exist_ok=True)

            # get labeled magnitudes of chosen signal
            series_df = creat_time_series(
                str(data_path),
                set_data_types([self.data_type]),
                [self.act],
                [self.data_params.activity_codes[self.act]]
            )

            data = []
            for act_code in self.data_params.activity_codes[self.act]:
                for subj_id in range(self.data_params.num_participants):
                    print(f"Activity: {self.act}; Act_code: {act_code}; Participant: {subj_id}")

                    # get time series for current participant and activity code
                    series = series_df.loc[
                        (series_df["id"] == subj_id) & (
                            series_df["trial"] == act_code),
                        [self.data_type]
                    ].values

                    cur_data = takens_traj(series, self.dim, self.max_len)
                    cur_num_traj = cur_data[0].shape[0]
                    cur_data = list(cur_data) + [[subj_id] * cur_num_traj]
                    data.append(cur_data)

            # save data as tensors
            any(
                starmap(
                    lambda data_list, i: torch.save(torch.from_numpy(np.concat(data_list)), self._data_files[i]),
                    list(zip(*data) | izip(range(3)))
                )
            )
        else:
            warnings.warn(f"Dir {save_dir} already exist, don't rebuild trajectires.")

        # load files in the desired mmap mode
        self._data = list(
            self._data_files |
            select(lambda data_f: torch.load(data_f, weights_only=True, mmap=self._mmap))
        )

        # count number of traj slices
        self._num_trajs = len(self._data[0])

    def __len__(self):
        return self._num_trajs

    def __getitem__(self, index):
        return tuple(
            self._data |
            select(lambda data_tensor: data_tensor[index])
        )
    
    def __getitems__(self, indexes):
        return self.__getitem__(indexes)


class ActivityDataModule(L.LightningDataModule):
    def __init__(
        self,
        act: str,
        dim: int,
        max_len: int,
        data_path: Path,
        save_dir: Path,            # storage file
        mmap: Optional[bool],
        data_type: str = "rotationRate",
        batch_size: int = 32,
        test_ratio: float = 0.2
    ):
        self.batch_size = batch_size
        self.test_ratio = test_ratio

        self.dataset_kwargs = deepcopy(locals().keys())
        self.dataset_kwargs = list(
            filter(lambda key: key not in ("batch_size", "test_ratio"), self.dataset_kwargs)
        )
        self.dataset_kwargs = {deepcopy(locals()[key]) for key in self.dataset_kwargs}

    def prepare_data(self):
        # build trajectories for the first time
        ActivityTrajDataset(**self.dataset_kwargs)

    def setup(self, stage):
        # load ready trajectories
        # split them for dataloaders
        self.train_dataset, self.val_dataset = random_split(
            ActivityTrajDataset(**self.dataset_kwargs),
            [1 - self.test_ratio, self.test_ratio],
            torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False
        )
    
    def predict_dataloader(self):
        return self.val_dataloader()