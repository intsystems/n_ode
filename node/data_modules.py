from typing import Optional
from pathlib import Path
import pickle
from omegaconf import OmegaConf
from pydantic import BaseModel

from pipe import select, izip, where
from itertools import starmap

import numpy as np
from pandas import DataFrame

import torch
from torch.utils.data import Dataset, Subset, DataLoader

import lightning as L

from .raw_data_loading import create_time_series, set_data_types
from .traj_build import takens_traj

import warnings


def build_sliced_takens_trajs(
    act_code: int,
    subj_id: int,
    series_df: DataFrame,
    dim: int,
    max_len: int,
    data_type: str = "rotationRate"
) -> list[torch.Tensor]:
    # get time series for current participant and activity code
    series = series_df.loc[
        (series_df["id"] == subj_id) & (
            series_df["trial"] == act_code),
        [data_type]
    ].values

    cur_data = takens_traj(series, dim, max_len)
    cur_data = (
        torch.tensor(cur_data[0], dtype=torch.float32),
        torch.tensor(cur_data[1], dtype=torch.int)
    )
    cur_num_traj = cur_data[0].shape[0]
    cur_data = list(cur_data) + [torch.tensor([subj_id] * cur_num_traj)]

    return cur_data

def build_subj_act_trajs(
    act: str,
    subj_id: int,
    dim: int,
    max_len: int,
    data_dir: Path,
    data_type: str = "rotationRate"
) -> list[dict[str, torch.Tensor]]:
    """
    Returns:
        list[dict[str, torch.Tensor]]: sliced trajectories and their metainfo ("traj", "dur", "subj_id", "traj_num")
    """
    # load config file for data
    data_params = OmegaConf.load(data_dir / "dataset_params.yaml")

    # get labeled magnitudes of chosen signal
    series_df = create_time_series(
        str(data_dir),
        set_data_types([data_type]),
        [act],
        [data_params.activity_codes[act]]
    )

    data = []
    # initial trajectory number for slices
    traj_num = 0
    for act_code in data_params.activity_codes[act]:
        print(f"Activity: {act}; Act_code: {act_code}; Participant: {subj_id}")

        cur_data = build_sliced_takens_trajs(
            act_code,
            subj_id,
            series_df,
            dim,
            max_len,
            data_type
        )
        cur_num_traj = cur_data[0].shape[0]
        cur_data += [torch.tensor([traj_num] * cur_num_traj)]

        names = ["traj", "dur", "subj_id", "traj_num"]
        data.append({
            name: v for (name, v) in zip(names, cur_data)
        })

        traj_num += 1

    return data

def build_act_trajs(
    act: str,
    dim: int,
    max_len: int,
    data_dir: Path,
    data_type: str = "rotationRate"
) -> list[dict[str, torch.Tensor]]:
    """
    Returns:
        list[dict[str, torch.Tensor]]: sliced trajectories and their metainfo ("traj", "dur", "subj_id", "traj_num")
    """
    # load config file for data
    data_params = OmegaConf.load(data_dir / "dataset_params.yaml")

    # get labeled magnitudes of chosen signal
    series_df = create_time_series(
        str(data_dir),
        set_data_types([data_type]),
        [act],
        [data_params.activity_codes[act]]
    )

    data = []
    # initial trajectory number for slices
    traj_num = 0
    for act_code in data_params.activity_codes[act]:
        for subj_id in range(data_params.num_participants):
            print(f"Activity: {act}; Act_code: {act_code}; Participant: {subj_id}")

            cur_data = build_sliced_takens_trajs(
                act_code,
                subj_id,
                series_df,
                dim,
                max_len,
                data_type
            )
            cur_num_traj = cur_data[0].shape[0]
            cur_data += [torch.tensor([traj_num] * cur_num_traj)]

            names = ["traj", "dur", "subj_id", "traj_num"]
            data.append({
                name: v for (name, v) in zip(names, cur_data)
            })

            traj_num += 1

    return data


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
        data_type: str = "rotationRate"
    ):
        super().__init__()

        # save info about dataset + engineering staff
        self.act = act
        self.dim = dim
        self.max_len = max_len
        self.data_type = data_type
        self.save_dir = save_dir
        # load config file for data
        self.data_params = OmegaConf.load(data_path / "dataset_params.yaml")

        # all data artifacts to generate
        self._data_files = {
            name: save_dir / (name + ".pt")
            for name in ("trajs", "durations", "subjs", "traj_nums")
        }

        if self.save_dir.exists():
            raise ValueError("Save dir already exists.")
        
        # build trajectories
        print("Building trajectories")

        self.save_dir.mkdir(parents=True)

        # get labeled magnitudes of chosen signal
        series_df = create_time_series(
            str(data_path),
            set_data_types([self.data_type]),
            [self.act],
            [self.data_params.activity_codes[self.act]]
        )

        data = []
        # initial trajectory number for slices
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

                cur_data: tuple[np.ndarray] = takens_traj(series, self.dim, self.max_len)

                cur_num_traj = cur_data[0].shape[0]
                cur_data = list(cur_data) + [[subj_id] * cur_num_traj] + [[traj_num] * cur_num_traj]

                data.append(cur_data)
                traj_num += 1

        # save data as tensors on disk
        for data_component, component_name in zip(*data) | izip(self._data_files):
            concat_data_component = torch.from_numpy(np.concat(data_component))
            if concat_data_component.dtype is torch.float64:
                concat_data_component = concat_data_component.to(torch.float32)
            torch.save(concat_data_component, self._data_files[component_name])

        del data
        del series_df

        # load files
        self._data_components = {
            name: torch.load(path, weights_only=True)
            for name, path in self._data_files.items()
        }

        # count number of traj slices
        self._num_trajs = len(self._data_components["trajs"])

    def __len__(self):
        return self._num_trajs
    
    @property
    def num_trajs(self):
        """ returns total number of **full** trajectories in the dataset
        """
        return torch.load(self._data_files["traj_nums"], weights_only=True).max().item()

    def __getitem__(self, index) -> tuple[torch.Tensor]:
        return tuple(
            self._data_components.values() |
            select(lambda data_tensor: data_tensor[index])
        )
    
    def __getitems__(self, indexes) -> tuple[torch.Tensor]:
        return self.__getitem__(indexes)


class ActivityDataModule(L.LightningDataModule):
    def __init__(
        self,
        save_dir: Path,            # storage file
        act: Optional[str] = None,
        dim: Optional[int] = None,
        max_len: Optional[int] = None,
        data_path: Optional[Path] = None,
        data_type: Optional[str] = "rotationRate",
        batch_size: int = 32,
        test_ratio: float = 0.2,     # по полным траекториям
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
        save_dir = Path(self.dataset_kwargs["save_dir"])

        if not save_dir.exists():
            # build trajectories and save them on disk
            dataset = ActivityTrajDataset(**self.dataset_kwargs)
            with open(save_dir / "dataset.pkl", "wb") as f:
                pickle.dump(dataset, f)

    def setup(self, stage):
        # load trajectories from disk
        with open(Path(self.dataset_kwargs["save_dir"]) / "dataset.pkl", "rb") as f:
            self.dataset: ActivityTrajDataset = pickle.load(f)
        print("Num data points ~", len(self.dataset) * self.dataset.max_len)
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
