""" Script to launch classification with gaussian bayes.
    Editable.
"""
import argparse
import pickle
from pathlib import Path
from omegaconf import OmegaConf
from rich.console import Console

import pandas as pd

from sklearn.mixture import GaussianMixture

import wandb


console = Console()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("act_feat_matr_path", type=Path)
    parser.add_argument("gauss_config_path", type=Path)
    parser.add_argument("model_save_dir", type=Path)
    parser.add_argument("classify_save_dir", type=Path)
    args = parser.parse_args()

    gauss_config = OmegaConf.load(args.gauss_config_path)

    with open(args.act_feat_matr_path, "rb") as f:
        acts_train_test = pickle.load(f)
    acts = list(acts_train_test.keys())

    # do data transformations
    pass

    classifiers = {
        act: GaussianMixture(**dict(gauss_config))
        for act in acts
    }
    for act, classifier in classifiers.items():
        with console.status(f"Training model for {act}..."):
            classifier.fit(acts_train_test[act]["train"])
    with open(args.model_save_dir / "gauss.pkl", "wb") as f:
        pickle.dump(classifiers, f)

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
    save_dir = Path(args.classify_save_dir)
    result.to_csv(save_dir / "gauss.csv", index=False)
