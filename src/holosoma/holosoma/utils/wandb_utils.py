"""Shared controls for the data Holosoma sends to Weights & Biases."""

from __future__ import annotations

from typing import Any


# Keep W&B lean by default: explicit scalar/history logging is allowed, while
# file and media uploads must be opted into by a non-metrics-only logger.
_metrics_only = True


def configure_wandb_uploads(*, metrics_only: bool) -> None:
    """Set the process-wide W&B upload policy for the current training run."""

    global _metrics_only
    _metrics_only = metrics_only


def wandb_file_uploads_enabled() -> bool:
    """Return whether model, config, log, or media files may be sent to W&B."""

    return not _metrics_only


def metrics_only_wandb_settings(wandb_module: Any) -> Any:
    """Build W&B settings that retain chart metrics without auxiliary files.

    Console capture is what creates and streams ``output.log``.  The remaining
    switches suppress source, Git, machine metadata, and dependency snapshots;
    explicitly logged history and system-stat chart data remain enabled.
    """

    return wandb_module.Settings(
        console="off",
        disable_code=True,
        disable_git=True,
        save_code=False,
        x_disable_meta=True,
        x_save_requirements=False,
    )
