"""
Inference Engine and Evaluation Pipeline.

Runs CriticalityNet in no_grad + autocast mode on the test set,
computes all metrics, and exports to JSON.

Usage
-----
python -m evaluation.evaluator --config configs/default_config.yaml --checkpoint auto
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

from data.dataset import build_dataloaders
from models.criticality_net import CriticalityNet
from models.utils import discover_latest_checkpoint, load_checkpoint, set_seed
from evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class Evaluator:
    """
    Inference and evaluation pipeline for CriticalityNet.

    Parameters
    ----------
    cfg : dict
        Full configuration dictionary.
    checkpoint_path : str
        Path to checkpoint or 'auto' to discover latest.
    """

    def __init__(self, cfg: dict, checkpoint_path: str = "auto") -> None:
        self.cfg = cfg
        set_seed(cfg["project"]["seed"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CriticalityNet.from_config(cfg["model"]).to(self.device)

        # Load checkpoint
        ckpt_path = checkpoint_path
        if ckpt_path == "auto":
            ckpt_path = discover_latest_checkpoint(cfg["project"]["checkpoint_dir"])
            if ckpt_path is None:
                raise FileNotFoundError(
                    "No checkpoint found in: " + cfg["project"]["checkpoint_dir"]
                )

        load_checkpoint(ckpt_path, self.model, device=str(self.device))
        self.model.eval()
        self.use_amp = torch.cuda.is_available()

    def predict(
        self,
        loader: torch.utils.data.DataLoader,
    ) -> dict[str, np.ndarray]:
        """
        Run inference on a DataLoader.

        Parameters
        ----------
        loader : DataLoader

        Returns
        -------
        dict with keys: 'crit_preds', 'crit_labels', 'sigma_preds',
                        'sigma_targets', 'geno_preds', 'geno_labels'
        """
        crit_preds, crit_labels = [], []
        sigma_preds, sigma_targets = [], []
        geno_preds, geno_labels = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Inference", leave=False):
                spikes = batch["spikes"].to(self.device)

                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(spikes)

                # Criticality
                cp = outputs["criticality_logits"].argmax(dim=-1).cpu().numpy()
                crit_preds.extend(cp)
                crit_labels.extend(batch["criticality_label"].numpy())

                # Branching ratio
                if "branching_ratio" in outputs:
                    sp = outputs["branching_ratio"].squeeze(-1).cpu().numpy()
                    sigma_preds.extend(sp)
                    sigma_targets.extend(batch["true_sigma"].numpy())

                # Genotype
                if "genotype_logits" in outputs:
                    gp = outputs["genotype_logits"].argmax(dim=-1).cpu().numpy()
                    geno_preds.extend(gp)
                    geno_labels.extend(batch["genotype_id"].numpy())

        return {
            "crit_preds": np.array(crit_preds),
            "crit_labels": np.array(crit_labels),
            "sigma_preds": np.array(sigma_preds),
            "sigma_targets": np.array(sigma_targets),
            "geno_preds": np.array(geno_preds),
            "geno_labels": np.array(geno_labels),
        }

    def run_and_export(self) -> dict:
        """
        Run evaluation on test set and export metrics to JSON.

        Returns
        -------
        dict
            All computed metrics.
        """
        _, _, test_loader = build_dataloaders(self.cfg, seed=self.cfg["project"]["seed"])
        preds = self.predict(test_loader)
        metrics = compute_all_metrics(preds)

        # Export
        results_dir = self.cfg["project"]["results_dir"]
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        out_path = str(Path(results_dir) / "metrics.json")

        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2, default=float)

        logger.info("Metrics exported → %s", out_path)
        logger.info("Criticality Acc: %.3f | F1: %.3f | Genotype F1: %.3f",
                    metrics.get("criticality_accuracy", 0),
                    metrics.get("criticality_f1_macro", 0),
                    metrics.get("genotype_f1_macro", 0))
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="auto")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluator = Evaluator(cfg, checkpoint_path=args.checkpoint)
    evaluator.run_and_export()


if __name__ == "__main__":
    main()
