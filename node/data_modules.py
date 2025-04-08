from typing import Optional
from pathlib import Path
import shutil
from omegaconf import OmegaConf

from pipe import select, izip, where
from itertools import starmap

import numpy as np

import torch
from torch.utils.data import Dataset, Subset, DataLoader

import lightning as L

from .raw_data_loading import creat_time_series, set_data_types
from .traj_build import takens_traj

import warnings

class ActivityTrajDataset(Dataset):
    """ Build phase trajectories for given activity.
        Building is deterministic.
    """
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
            ("trajs.pt", "durations.pt", "subjs.pt", "traj_nums.pt") |
            select(lambda data_name: save_dir / data_name)
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
            # inital trajectory number for slices
            traj_num = 0
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
                    cur_data = list(cur_data) + [[subj_id] * cur_num_traj] + [[traj_num] * cur_num_traj]

                    data.append(cur_data)
                    traj_num += 1

            # save data as tensors on disk
            for data_component, i in zip(*data) | izip(range(len(self._data_files))):
                concat_data_component = torch.from_numpy(np.concat(data_component))
                if concat_data_component.dtype is torch.float64:
                    concat_data_component = concat_data_component.to(torch.float32)
                torch.save(concat_data_component, self._data_files[i])
        else:
            warnings.warn(f"Dir {save_dir} already exist, don't rebuild trajectires.")

        # load files in the desired mmap mode
        self._data_tuples = list(
            self._data_files |
            select(lambda data_f: torch.load(data_f, weights_only=True, mmap=self._mmap))
        )

        # count number of traj slices
        self._num_trajs = len(self._data_tuples[0])

    def __len__(self):
        return self._num_trajs
    
    @property
    def num_trajs(self):
        """ returns total number of **full** trajectories in the dataset
        """
        return torch.load(self._data_files[-1], weights_only=True, mmap=self._mmap)[-1].item()

    def __getitem__(self, index) -> tuple[torch.Tensor]:
        return tuple(
            self._data_tuples |
            select(lambda data_tensor: data_tensor[index])
        )
    
    def __getitems__(self, indexes) -> tuple[torch.Tensor]:
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
        test_ratio: float = 0.2     # по полным траекториям
    ):
        super().__init__()

        self.batch_size = batch_size
        self.test_ratio = test_ratio

        dataset_kwargs_keys = list(
            filter(lambda k: k not in {"batch_size", "test_ratio", "self", "__class__"}, locals().keys())
        )
        self.dataset_kwargs = {
            key: locals()[key]
            for key in dataset_kwargs_keys
        }

    def prepare_data(self):
        # build trajectories and save them on disk
        ActivityTrajDataset(**self.dataset_kwargs)

    def setup(self, stage):
        # load trajectories from disk
        self.dataset = ActivityTrajDataset(**self.dataset_kwargs)
        num_trajs = self.dataset.num_trajs
        # split FULL trajectories according to "test_ratio"
        split_index = max(
            range(len(self.dataset)) |
            where(lambda i: self.dataset[i][-1] < int((1 - self.test_ratio) * num_trajs))
        )
        self.train_dataset = Subset(self.dataset, list(range(split_index)))
        self.val_dataset = Subset(self.dataset, list(range(split_index, len(self.dataset))))

    def train_dataloader(self):
        """As dataset can give already correct batched tuples with __getitems__,
            identical collate_fn is used.
        """
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            collate_fn=lambda x: x
        )

    def val_dataloader(self):
        """As dataset can give already correct batched tuples with __getitems__,
            identical collate_fn is used.
        """
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            collate_fn=lambda x: x
        )

    def predict_dataloader(self):
        return self.val_dataloader()

    def teardown(self, stage):
        shutil.rmtree(self.dataset_kwargs["save_dir"])
