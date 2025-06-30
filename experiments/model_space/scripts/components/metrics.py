from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score


def compute_agragate_metrics(y_true, y_pred, method: str):
    return pd.DataFrame(
        {
            "method": method,
            "accuracy": accuracy_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred, average="macro"),
            "f1": f1_score(y_true, y_pred, average="macro")
        },
        index=[0]
    )


def make_recall_hist(
    y_true: pd.Series,
    y_pred: pd.Series,
    method: str
):
    acts = y_true.unique().tolist()
    recall_hist = []
    for act in acts:
        tp = y_pred[(y_pred == act) & (y_true == act)].size
        tp_fn = y_true[y_true == act].size
        recall_hist.append([act, (tp / tp_fn) if tp_fn != 0 else 0., method])
    recall_hist = pd.DataFrame(recall_hist, columns=["act", "recall", "method"])

    return recall_hist
