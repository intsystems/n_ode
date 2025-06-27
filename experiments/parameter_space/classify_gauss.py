""" Script to launch trajectories classification with gaussian bayes.
    Editable.
"""
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from rich.console import Console

import pandas as pd

from sklearn.mixture import GaussianMixture

import wandb

from components.field_model import MyVectorField
from components.feature import get_acts_feat_splitted


console = Console()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints_dir", type=Path)
    parser.add_argument("split_config_path", type=Path)
    parser.add_argument("gauss_config_path", type=Path)
    parser.add_argument(
        "shared_train_config_path", type=Path, 
        help="Needed to reconstuct vf and extract params"
    )
    args = parser.parse_args()

    gauss_config = OmegaConf.load(args.gauss_config_path)
    split_config = OmegaConf.load(args.split_config_path)
    train_config = OmegaConf.load(args.shared_train_config_path)

    acts_train_test = get_acts_feat_splitted(
        args.checkpoints_dir,
        MyVectorField(**dict(train_config.vf)),
        **dict(split_config)
    )
    acts = list(acts_train_test.keys())

    # do data transformations
    pass

    classifiers = {
        act: GaussianMixture(**dict(gauss_config))
        for act in acts
    }
    for act, classifier in classifiers.items():
        classifier.fit(acts_train_test[act]["train"])

    # assume each class has equal prior prob
    # and classify
    result = []
    for act in acts:
        acts_probs = {
            pred_act: classifier.predict_proba(acts_train_test[act]["val"])\
                                .dot(classifier.weights_)
            for pred_act, classifier in classifiers.items()
        }
        acts_probs = pd.DataFrame(acts_probs)
        preds = acts_probs.idxmax(axis=1)
        result.append(pd.DataFrame({
            "pred": preds,
            "true": [act] * preds.size
        }))
    result = pd.concat(result)
    save_dir = Path("classify")
    save_dir.mkdir(exist_ok=True)
    result.to_csv(save_dir / "gauss.csv", index=False)

    accuracy = (result["pred"] == result["true"]).mean()
    console.print("Accuracy = ", accuracy)
