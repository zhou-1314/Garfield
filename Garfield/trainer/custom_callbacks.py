"""
Custom PyTorch Lightning callbacks for Garfield model training.

This module provides custom callbacks including a notebook-style progress bar
that mimics the original Garfield trainer logging format.
"""
import sys
from typing import Any, Dict, Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


def _is_notebook() -> bool:
    """
    Detect if code is running in a Jupyter notebook environment.

    Returns
    -------
    bool
        True if running in a notebook, False otherwise.
    """
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            return True
    except (ImportError, AttributeError):
        pass
    return False


def _print_notebook_progress_bar(
    epoch: int,
    n_epochs: int,
    metrics: Dict[str, float],
    prefix: str = "",
    decimals: int = 1,
    length: int = 20,
    fill: str = "█",
):
    """
    Print a notebook-style progress bar with metrics.

    This function implements the original Garfield trainer progress bar format,
    displaying a visual progress bar followed by training metrics.

    Parameters
    ----------
    epoch : int
        Current epoch number (1-indexed).
    n_epochs : int
        Total number of epochs.
    metrics : Dict[str, float]
        Dictionary of metric names and values to display.
    prefix : str, optional
        String to display before the progress bar.
    decimals : int, optional
        Number of decimal places for percentage display.
    length : int, optional
        Character length of the progress bar.
    fill : str, optional
        Character to use for the filled portion of the bar.
    """
    # Calculate percentage and bar fill
    percent = ("{0:." + str(decimals) + "f}").format(100 * (epoch / float(n_epochs)))
    filled_len = int(length * epoch // n_epochs)
    bar = fill * filled_len + "-" * (length - filled_len)

    # Format metrics string
    metrics_str = ""
    for key, value in metrics.items():
        metrics_str += f"{key}: {value:.4f}; "
    if metrics_str:
        metrics_str = metrics_str[:-2]  # Remove trailing "; "

    # Print progress bar and metrics
    sys.stdout.write(f"\r{prefix} |{bar}| {percent}% {metrics_str}")

    # Add newline on final epoch
    if epoch == n_epochs:
        sys.stdout.write("\n")

    sys.stdout.flush()


class NotebookProgressBar(Callback):
    """
    Custom PyTorch Lightning callback that provides notebook-style progress logging.

    This callback mimics the original Garfield trainer's logging format, displaying
    a visual progress bar with metrics inline. It's designed for use in Jupyter
    notebooks to provide a clean, compact training progress display.

    The callback displays:
    - Visual progress bar showing training completion
    - Epoch number and total epochs
    - All logged metrics (train/val losses and scores)

    Parameters
    ----------
    n_epochs : int
        Total number of training epochs.
    verbose : bool, optional
        If True, display all logged metrics. If False, only show main losses.
    """

    def __init__(self, n_epochs: int, verbose: bool = False):
        super().__init__()
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.epoch_metrics = {}

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called at the end of each training epoch.

        Collects metrics and displays the progress bar.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer instance.
        pl_module : pl.LightningModule
            Lightning module being trained.
        """
        # Get current epoch (1-indexed for display)
        current_epoch = trainer.current_epoch + 1

        # Collect metrics from trainer
        metrics = {}

        # Get logged metrics from trainer
        logged_metrics = trainer.callback_metrics

        # Filter metrics based on verbose setting
        for key, value in logged_metrics.items():
            # Skip some internal metrics
            if key in ['epoch', 'step']:
                continue

            # Convert tensor to float
            if hasattr(value, 'item'):
                value = value.item()

            # If not verbose, only show main losses
            if not self.verbose:
                if any(main in key for main in ['global_loss', 'optim_loss']):
                    metrics[key] = value
            else:
                metrics[key] = value

        # Display progress bar
        _print_notebook_progress_bar(
            epoch=current_epoch,
            n_epochs=self.n_epochs,
            metrics=metrics,
            prefix=f"Epoch {current_epoch}/{self.n_epochs}",
        )

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called at the end of training.

        Ensures a newline is printed after the final progress bar.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer instance.
        pl_module : pl.LightningModule
            Lightning module being trained.
        """
        # Ensure we end with a newline
        sys.stdout.write("\n")
        sys.stdout.flush()
