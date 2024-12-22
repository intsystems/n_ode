from pathlib import Path
from shutil import rmtree
import yaml

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
import scipy.linalg as linalg

from .raw_data_loading import creat_time_series, set_data_types


def slice_trajectory(traj_matrix: np.ndarray, traj_len: int) -> tuple[np.ndarray]:
    """ function for slicing phase trajectory on patches with traj_len length
    """
    traj_dim = traj_matrix.shape[1]
    num_slices = traj_matrix.shape[0] // traj_len
    traj_residue_len = traj_matrix.shape[0] % traj_len

    # compose full length batches
    if num_slices > 0:
        batches = traj_matrix[:num_slices * traj_len].reshape((num_slices, traj_len, traj_dim))
    else:
        batches = np.empty((0, traj_len, traj_dim))
    # compose residual batches
    pad_len = traj_len - traj_residue_len
    if pad_len > 0:
        residue_batch = np.concat(
            [traj_matrix[num_slices * traj_len:], np.zeros((pad_len, traj_dim))]
        )
        residue_batch = np.expand_dims(residue_batch, axis=0)
        batches = np.concat([batches, residue_batch])

    # compute batches durations
    durations = [traj_len for _ in range(num_slices)] + [traj_residue_len]
    durations = np.array(durations)

    return batches, durations


def normalize_trajectories(dataset: Dataset) -> Dataset:
    """ Normalize trajectories (use for train and test simultainiously)
    """
    traj: torch.Tensor = dataset.tensors[0].detach()
    traj = (traj - traj.mean(dim=0)) / traj.std(dim=0)

    return TensorDataset(traj, dataset.tensors[1].detach())


def make_activity_dataset(activity: str, traj_dir: Path) -> Dataset:
    act_files = list(traj_dir.glob(f"{activity}*[!(_duration)].pt"))
    traj_list = list(map(
        torch.load,
        act_files
    ))

    dur_files = list(traj_dir.glob(f"{activity}*_duration.pt"))
    dur_list = list(map(
        torch.load,
        dur_files
    ))

    # merge trajectories
    trajectories = torch.concat(traj_list)
    durations = torch.concat(dur_list)

    return TensorDataset(trajectories, durations)


def make_trajectories(config: dict, data_path: Path, traj_dir: Path = "trajectories/"):
    """Creates trajectories as torch.Tensors for each activity, activity code, participant.

    Args:
        config (dict): _description_
        data_path (Path): _description_
        traj_dir (Path, optional): dir to save trajectories. Defaults to "trajectories/".
    """
    print("Building phase trajectories...")

    # load config files for data
    with open(data_path / "dataset_params.yaml") as f2:
        data_params = yaml.full_load(f2)

    # make dir for trajectories
    if traj_dir.exists():
        rmtree(traj_dir)
    traj_dir.mkdir()

    for activity, act_codes in data_params["activity_codes"].items():
        # get labeled magnitudes of chosen signal
        series_df = creat_time_series(
            str(data_path),
            set_data_types([config["data_type"]]),
            [activity],
            [act_codes]
        )

        for act_code in act_codes:
            for partic_id in range(data_params["num_participants"]):
                print(f"Activity: {activity}; Act_code: {act_code}; Participant: {partic_id}")

                # get time series for current participant and activity code
                series = series_df.loc[
                    (series_df["id"] == partic_id) & (
                        series_df["trial"] == act_code),
                    [config["data_type"]]
                ].values

                # build trajectory matrix of the series
                traj_matrix = linalg.hankel(
                    series[:config["trajectory_dim"]
                           ], series[config["trajectory_dim"] - 1:]
                ).astype(np.float32)
                # transpose matrix so time axis = 0
                traj_matrix = traj_matrix.T
                # get batched trajectory
                traj_batch, duration = slice_trajectory(traj_matrix, config["trajectory_len"])

                traj_batch = torch.from_numpy(traj_batch.astype(np.float32))
                duration = torch.from_numpy(duration.astype(np.int64))

                # save dataset on disk
                torch.save(
                    traj_batch,
                    traj_dir / f"{activity}_{act_code}_{partic_id}.pt"
                )
                torch.save(
                    duration,
                    traj_dir / f"{activity}_{act_code}_{partic_id}_duration.pt"
                )


# tests
# # load config files for data and pipeline
# with open("../experiments/hypothesis/config.yaml", "r") as f1:
#     config = yaml.full_load(f1)

# make_trajectories(
#     config,
#     Path("../data"),
#     Path("../data/trajectories")
# )

# dataset = make_activity_dataset("std", Path("../data/trajectories"))
