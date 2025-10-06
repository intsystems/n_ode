""" Merge shared and activity-specific configs for data and train
"""
import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("act", type=str)
    args = parser.parse_args()

    CONFIG_TYPES = ["data", "train"]
    for config_type in CONFIG_TYPES:
        config: DictConfig = DictConfig({})
        # load activity-spec data conf
        spec_conf = Path(f"config/{args.act}/{config_type}.yaml")
        if spec_conf.exists():
            config = OmegaConf.merge(
                config, OmegaConf.load(spec_conf)
            )
        # load shared data conf
        shared_conf = Path(f"config/{config_type}.yaml")
        if shared_conf.exists():
            config = OmegaConf.merge(
                config, OmegaConf.load(shared_conf)
            )
        OmegaConf.save(config, f"tmp/config/{args.act}/{config_type}.yaml")
