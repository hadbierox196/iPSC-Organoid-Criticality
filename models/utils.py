"""
Model utility functions: checkpointing, weight init, memory summary, seed locking.
"""

import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Lock all random seeds for full reproducibility.

    Parameters
    ----------
    seed : int
        Global random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Global seed set to %d.", seed)


def count_parameters(model: nn.Module) -> int:
    """
    Count total trainable parameters.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    int
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def memory_summary(model: nn.Module, input_shape: tuple[int, ...]) -> dict[str, Any]:
    """
    Compute approximate model memory footprint.

    Parameters
    ----------
    model : nn.Module
    input_shape : tuple
        Input tensor shape (excluding batch dim).

    Returns
    -------
    dict
        'n_params', 'param_mb', 'input_mb', 'total_mb' estimates.
    """
    n_params = count_parameters(model)
    param_mb = n_params * 4 / 1e6   # float32 = 4 bytes
    input_elems = 1
    for d in input_shape:
        input_elems *= d
    input_mb = input_elems * 4 / 1e6

    summary = {
        "n_params": n_params,
        "param_mb": round(param_mb, 2),
        "input_mb_per_sample": round(input_mb, 4),
        "estimated_total_mb": round(param_mb + input_mb * 32, 2),  # rough est. for batch=32
    }
    logger.info(
        "Model: %d params (%.1f MB param + ~%.1f MB batch-32 activations)",
        n_params, param_mb, input_mb * 32,
    )
    assert param_mb < 500, f"Model too large for Colab: {param_mb:.1f} MB params."
    return summary


def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    metrics: dict,
    cfg: dict,
    max_checkpoints: int = 3,
) -> None:
    """
    Save training checkpoint to disk (Google Drive recommended).

    Parameters
    ----------
    path : str
        Full path to checkpoint file (.pth).
    epoch : int
        Current epoch number.
    model : nn.Module
    optimizer : Optimizer
    scheduler : LR scheduler
    metrics : dict
        Dict of current validation metrics.
    cfg : dict
        Full config dict (for reproducibility).
    max_checkpoints : int
        Keep only the latest N checkpoints (older ones deleted).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, "state_dict") else {},
        "metrics": metrics,
        "cfg": cfg,
        "timestamp": time.time(),
    }
    torch.save(state, path)
    logger.info("Checkpoint saved → %s (epoch %d)", path, epoch)

    # Prune old checkpoints
    ckpt_dir = Path(path).parent
    ckpt_pattern = "checkpoint_epoch_*.pth"
    existing = sorted(ckpt_dir.glob(ckpt_pattern))
    if len(existing) > max_checkpoints:
        for old in existing[:-max_checkpoints]:
            old.unlink()
            logger.debug("Deleted old checkpoint: %s", old)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    device: str = "cpu",
) -> dict:
    """
    Load a training checkpoint, restoring model (and optionally optimizer) state.

    Parameters
    ----------
    path : str
        Path to .pth checkpoint file. Use 'auto' to discover latest in checkpoint_dir.
    model : nn.Module
    optimizer : Optimizer, optional
    scheduler : optional
    device : str

    Returns
    -------
    dict
        Full checkpoint state including epoch and metrics.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    logger.info(
        "Loaded checkpoint from epoch %d (metrics: %s)",
        state["epoch"], state.get("metrics", {}),
    )
    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in state and state["scheduler_state_dict"]:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    return state


def discover_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint file in a directory.

    Parameters
    ----------
    checkpoint_dir : str

    Returns
    -------
    str or None
        Path to latest checkpoint, or None if none found.
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"))
    if not checkpoints:
        return None
    latest = str(checkpoints[-1])
    logger.info("Discovered latest checkpoint: %s", latest)
    return latest
