""" Script to launch datasets building.
    Operates in snakemake ecosystem.
"""
from pathlib import Path
from snakemake.script import snakemake
from omegaconf import OmegaConf, DictConfig

import torch

from node.data_modules import build_sliced_takens_trajs


if __name__ == "__main__":
    # load config file
    data_config: DictConfig = OmegaConf.merge(
        OmegaConf.load(snakemake.input["config_path"]),
        OmegaConf.load(snakemake.input["shared_config_path"])
    )

    # build trajectories
    sliced_trajs = build_sliced_takens_trajs(
        data_dir=Path(snakemake.params.RAW_DATA_DIR),
        **dict(data_config.data)
    )
    # save trajectories in separate dirs
    for traj_dict in sliced_trajs:
        subj_id = traj_dict["subj_id"][0].item()
        traj_num = traj_dict["traj_num"][0].item()

        save_dir = Path("tmp/" + snakemake.params.DATASET_DIR_ROOT) / \
            f"{data_config.data.act}/subj_{subj_id}/traj_{traj_num}"
        save_dir.mkdir(exist_ok=True, parents=True)

        for name, data in traj_dict.items():
            torch.save(data, save_dir / f"{name}.pt")
