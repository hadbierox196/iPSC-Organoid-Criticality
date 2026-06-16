"""
Main Training Loop for CriticalityNet.

Features
--------
- YAML config parsing (runtime, no code changes)
- Automatic checkpoint discovery + resume on Google Drive
- Mixed-precision (torch.cuda.amp)
- Gradient accumulation
- Early stopping on val F1
- Estimated wall-time printouts (12-hour Colab guard)
- TensorBoard logging
- tqdm(leave=False) to prevent log overflow

Usage
-----
python -m training.trainer --config configs/default_config.yaml
python -m training.trainer --config configs/default_config.yaml --dry_run --max_epochs 2
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import build_dataloaders
from data.downloader import generate_synthetic_dataset
from models.criticality_net import CriticalityNet
from models.utils import (
    discover_latest_checkpoint,
    load_checkpoint,
    memory_summary,
    save_checkpoint,
    set_seed,
)
from training.losses import CriticalityLoss

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Maximum safe wall-time before warning (Colab limit = 12h)
COLAB_WALL_TIME_LIMIT_S = 11.5 * 3600


class EarlyStopping:
    """Monitors a validation metric and stops training when no improvement."""

    def __init__(self, patience: int = 15, mode: str = "max", min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        improved = (score > self.best_score + self.min_delta) if self.mode == "max" \
            else (score < self.best_score - self.min_delta)
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    Full training orchestrator for CriticalityNet.

    Parameters
    ----------
    cfg : dict
        Full configuration dictionary.
    dry_run : bool
        If True, only run max_epochs epochs (for CI testing).
    max_epochs : int
        Used with dry_run only.
    """

    def __init__(self, cfg: dict, dry_run: bool = False, max_epochs: int = 2) -> None:
        self.cfg = cfg
        self.dry_run = dry_run
        self.max_epochs_override = max_epochs if dry_run else None

        set_seed(cfg["project"]["seed"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device: %s", self.device)

        # Directories (Google Drive)
        self.ckpt_dir = cfg["project"]["checkpoint_dir"]
        self.log_dir = cfg["project"]["log_dir"]
        Path(self.ckpt_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # Model
        self.model = CriticalityNet.from_config(cfg["model"]).to(self.device)
        memory_summary(
            self.model,
            (cfg["model"]["n_electrodes"], cfg["model"]["n_time_bins"]),
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
        )

        # LR Scheduler (cosine with warmup)
        n_epochs = self.max_epochs_override or cfg["training"]["n_epochs"]
        warmup = cfg["training"].get("warmup_epochs", 5)
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.01, total_iters=warmup
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=max(1, n_epochs - warmup)
                ),
            ],
            milestones=[warmup],
        )

        # Loss
        loss_cfg = cfg["training"]["loss_weights"]
        self.criterion = CriticalityLoss(
            w_criticality=loss_cfg.get("criticality_ce", 1.0),
            w_sigma=loss_cfg.get("branching_ratio_mse", 0.5),
            w_genotype=loss_cfg.get("genotype_ce", 0.3),
            w_consistency=loss_cfg.get("consistency", 0.2),
        )

        # AMP
        self.use_amp = cfg["training"].get("mixed_precision", True) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # Grad accumulation
        self.accum_steps = cfg["training"].get("gradient_accumulation_steps", 4)

        # Early stopping
        self.early_stopper = EarlyStopping(
            patience=cfg["training"].get("early_stopping_patience", 15),
            mode="max",
        )

        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # State
        self.start_epoch = 0
        self.best_val_f1 = 0.0
        self.train_start_time = time.time()

    def _prepare_data(self) -> None:
        """Build or load dataset and create DataLoaders."""
        import os
        processed = self.cfg["data"]["processed_path"]
        if not Path(processed).exists():
            logger.info("Processed dataset not found. Generating synthetic data...")
            raw_path = self.cfg["data"]["local_path"]
            if not Path(raw_path).exists():
                sc = self.cfg["data"]["synthetic"]
                generate_synthetic_dataset(
                    output_path=raw_path,
                    n_organoids_per_genotype=sc.get("n_organoids_per_genotype", 4),
                    recording_duration_s=sc.get("recording_duration_s", 120.0),
                    n_electrodes=self.cfg["data"]["n_electrodes"],
                    sampling_rate=self.cfg["data"]["sampling_rate"],
                    seed=self.cfg["project"]["seed"],
                )
            # Copy raw to processed (raw is already in spike-train format)
            import shutil
            Path(processed).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw_path, processed)

        self.train_loader, self.val_loader, self.test_loader = build_dataloaders(
            self.cfg, seed=self.cfg["project"]["seed"]
        )
        logger.info(
            "DataLoaders: train=%d, val=%d, test=%d batches.",
            len(self.train_loader), len(self.val_loader), len(self.test_loader),
        )

    def _try_resume(self) -> None:
        """Attempt to resume from the latest checkpoint."""
        latest = discover_latest_checkpoint(self.ckpt_dir)
        if latest:
            state = load_checkpoint(
                latest, self.model, self.optimizer, self.scheduler,
                device=str(self.device),
            )
            self.start_epoch = state["epoch"] + 1
            self.best_val_f1 = state["metrics"].get("val_f1", 0.0)
            logger.info("Resuming from epoch %d.", self.start_epoch)
        else:
            logger.info("No checkpoint found — starting from scratch.")

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_correct = 0
        n_total = 0
        self.optimizer.zero_grad()

        with tqdm(self.train_loader, desc=f"Train E{epoch}", leave=False) as pbar:
            for step, batch in enumerate(pbar):
                spikes = batch["spikes"].to(self.device)
                crit_labels = batch["criticality_label"].to(self.device)
                sigma_targets = batch["true_sigma"].to(self.device)
                geno_labels = batch["genotype_id"].to(self.device)

                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(spikes)
                    loss, loss_dict = self.criterion(outputs, crit_labels, sigma_targets, geno_labels)
                    loss = loss / self.accum_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                # Accuracy
                preds = outputs["criticality_logits"].argmax(dim=-1)
                n_correct += (preds == crit_labels).sum().item()
                n_total += len(crit_labels)
                total_loss += loss.item() * self.accum_steps

                pbar.set_postfix(loss=f"{total_loss/(step+1):.4f}",
                                 acc=f"{n_correct/max(n_total,1):.3f}")

        return {
            "train_loss": total_loss / max(len(self.train_loader), 1),
            "train_acc": n_correct / max(n_total, 1),
        }

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict[str, float]:
        """Run one validation epoch."""
        from sklearn.metrics import f1_score

        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with tqdm(self.val_loader, desc=f"Val   E{epoch}", leave=False) as pbar:
            for batch in pbar:
                spikes = batch["spikes"].to(self.device)
                crit_labels = batch["criticality_label"].to(self.device)
                sigma_targets = batch["true_sigma"].to(self.device)
                geno_labels = batch["genotype_id"].to(self.device)

                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(spikes)
                    loss, _ = self.criterion(outputs, crit_labels, sigma_targets, geno_labels)

                total_loss += loss.item()
                preds = outputs["criticality_logits"].argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(crit_labels.cpu().numpy())

        val_loss = total_loss / max(len(self.val_loader), 1)
        import numpy as np
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        return {"val_loss": val_loss, "val_f1": float(val_f1)}

    def train(self) -> None:
        """Full training loop with checkpointing, early stopping, and wall-time guard."""
        self._prepare_data()
        self._try_resume()

        n_epochs = self.max_epochs_override or self.cfg["training"]["n_epochs"]
        logger.info(
            "Starting training: epochs %d→%d (device=%s, AMP=%s, accum=%d)",
            self.start_epoch, n_epochs, self.device, self.use_amp, self.accum_steps,
        )

        for epoch in range(self.start_epoch, n_epochs):
            # ── Colab 12-hour wall-time guard ─────────────────────────────
            elapsed = time.time() - self.train_start_time
            est_per_epoch = elapsed / max(epoch - self.start_epoch, 1)
            est_remaining = est_per_epoch * (n_epochs - epoch)
            logger.info(
                "Epoch %d/%d | Elapsed: %.1fh | Est. remaining: %.1fh",
                epoch, n_epochs, elapsed / 3600, est_remaining / 3600,
            )
            if elapsed + est_per_epoch > COLAB_WALL_TIME_LIMIT_S:
                logger.warning(
                    "⚠️  Approaching Colab 12-hour limit! Saving checkpoint and stopping."
                )
                self._save(epoch - 1, {"val_f1": self.best_val_f1})
                break

            train_metrics = self._train_epoch(epoch)
            val_metrics = self._val_epoch(epoch)
            self.scheduler.step()

            metrics = {**train_metrics, **val_metrics, "lr": self.scheduler.get_last_lr()[0]}
            logger.info(
                "E%d: train_loss=%.4f | train_acc=%.3f | val_loss=%.4f | val_f1=%.3f",
                epoch, metrics["train_loss"], metrics["train_acc"],
                metrics["val_loss"], metrics["val_f1"],
            )

            # TensorBoard
            for key, val in metrics.items():
                self.writer.add_scalar(f"metrics/{key}", val, epoch)

            # Best model
            if val_metrics["val_f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["val_f1"]
                self._save(epoch, metrics, suffix="best")

            # Periodic checkpoint
            if (epoch + 1) % self.cfg["training"].get("save_every_n_epochs", 5) == 0:
                self._save(epoch, metrics)

            # Early stopping
            if self.early_stopper(val_metrics["val_f1"]):
                logger.info("Early stopping triggered at epoch %d.", epoch)
                self._save(epoch, metrics, suffix="early_stop")
                break

        self.writer.close()
        logger.info("Training complete. Best val_f1: %.4f", self.best_val_f1)

    def _save(self, epoch: int, metrics: dict, suffix: str = "") -> None:
        name = f"checkpoint_epoch_{epoch:04d}{('_' + suffix) if suffix else ''}.pth"
        path = str(Path(self.ckpt_dir) / name)
        save_checkpoint(
            path, epoch, self.model, self.optimizer, self.scheduler,
            metrics, self.cfg,
            max_checkpoints=self.cfg["training"].get("max_checkpoints_to_keep", 3),
        )


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--dry_run", action="store_true", help="Run 2-epoch CI test.")
    parser.add_argument("--max_epochs", type=int, default=2, help="Epochs for dry_run.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    trainer = Trainer(cfg, dry_run=args.dry_run, max_epochs=args.max_epochs)
    trainer.train()


if __name__ == "__main__":
    main()
