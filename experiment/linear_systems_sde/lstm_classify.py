import os
import argparse
from pathlib import Path
from omegaconf import OmegaConf

import pandas as pd
import statsmodels.api as sm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchdiffeq import odeint
from torchmetrics.classification import MulticlassAccuracy
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from fields import matched_spectrum_trio, simulate_sde_paths

SEED = 847
NUM_SAMPLES = 1000
d = 3
VAL_FRAC = 0.1
TEST_FRAC = 0.1
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LR = 1e-3
MAX_EPOCHS = 50


class LSTMClassifier(LightningModule):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int,
        num_classes: int, lr: float,
        traj_mean: torch.Tensor, traj_std: torch.Tensor,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
        )
        self.head = nn.Linear(hidden_size, num_classes)
        self.lr = lr
        self.register_buffer("traj_mean", traj_mean)
        self.register_buffer("traj_std", traj_std)

        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

        self.save_hyperparameters(ignore=["traj_mean", "traj_std"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.traj_mean) / self.traj_std
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

    def _step(self, batch, acc_metric, stage: str):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc_metric.update(logits.argmax(dim=-1), y)
        loss_on_step = True if stage == "train" else False
        self.log(f"{stage}_loss", loss, on_step=loss_on_step, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", acc_metric, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, self.train_acc, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, self.val_acc, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, self.test_acc, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sigma", type=float)
    args = parser.parse_args()
    config = OmegaConf.load("experiment/linear_systems_sde/config.yaml")
    config.x0_sigma = 3.
    config.obs_sigma = args.sigma  # swept observation noise; sigma_proc stays fixed from config
    seed_everything(SEED)

    def generate_dataset() -> tuple[torch.Tensor, torch.Tensor]:
        fields = matched_spectrum_trio(spacing=1.0)
        t_mesh = torch.arange(config.traj_len) * config.dt

        trajs, labels = [], []
        for label, field in enumerate(fields):
            x0 = torch.randn((NUM_SAMPLES, d)) * config.x0_sigma
            traj = simulate_sde_paths(field, x0, t_mesh, config.sigma_proc)[1:]
            traj = traj + torch.randn_like(traj) * config.obs_sigma
            traj = traj.transpose(0, 1).to(torch.float32)
            trajs.append(traj)
            labels.append(torch.full((NUM_SAMPLES, ), label, dtype=torch.long))

        return torch.cat(trajs, dim=0), torch.cat(labels, dim=0)
    trajs, labels = generate_dataset()
    full_dataset = TensorDataset(trajs, labels)

    n_total = len(full_dataset)
    n_test = int(n_total * TEST_FRAC)
    n_val = int(n_total * VAL_FRAC)
    n_train = n_total - n_val - n_test
    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_trajs = trajs[train_set.indices]
    traj_mean = train_trajs.reshape(-1, d).mean(dim=0)
    traj_std = train_trajs.reshape(-1, d).std(dim=0).clamp_min(1e-6)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=2)

    model = LSTMClassifier(
        input_size=d, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
        num_classes=3, lr=LR,
        traj_mean=traj_mean, traj_std=traj_std,
    )

    save_dir = Path(config.results_dir) / "lstm"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = MLFlowLogger(
        experiment_name="linear_systems_sde",
        tracking_uri=config.tracking_uri,
        run_name="lstm_classify",
    )
    logger.log_hyperparams({"sigma": args.sigma})

    checkpoint = ModelCheckpoint(
        dirpath=save_dir, filename="best", monitor="val_acc", mode="max", save_top_k=1,
    )
    # early_stop = EarlyStopping(monitor="val_acc", mode="max", patience=20)

    trainer = Trainer(
        accelerator="cpu",
        max_epochs=MAX_EPOCHS,
        logger=logger,
        callbacks=[checkpoint],
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, val_loader)
    test_metrics = trainer.test(model, test_loader, ckpt_path="best")
    acc = test_metrics[0]["test_acc"]
    print("Test accuracy:", acc, "sigma", args.sigma)

    ci_low, ci_high = sm.stats.proportion_confint(
        int(acc * NUM_SAMPLES * TEST_FRAC), int(NUM_SAMPLES * TEST_FRAC), method="wilson"
    )
    df = pd.DataFrame({"sigma": [args.sigma], "accuracy": [acc], "ci_high": [ci_high], "ci_low": [ci_low]})
    df.to_csv(os.path.join(config.results_dir, "lstm", f"sigma_{args.sigma:.5f}.csv"))
