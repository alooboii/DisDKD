import csv
from pathlib import Path


class LossTracker:
    """Tracks and logs all training/validation losses and metrics."""

    def __init__(self, log_path: str, method: str):
        self.log_path = Path(log_path)
        self.method = method
        self.log_data = []

        # Initialize CSV with headers based on method
        if method == "DisDKD":
            headers = [
                "epoch",
                "disdkd_phase",  # 1, 2, or 3
                "split",  # train or val
                "disc_loss",  # Phase 1: discriminator BCE loss
                "disc_acc",  # Phase 1: discriminator accuracy (%)
                "adv_loss",  # Phase 2: adversarial loss
                "fool_rate",  # Phase 2: fool rate (%)
                "ce_loss",  # Phase 3: cross-entropy loss
                "dkd_loss",  # Phase 3: DKD loss (TCKD + NCKD)
                "total_loss",  # Total loss for the phase
                "accuracy",  # Classification accuracy (%)
            ]
        else:
            headers = ["epoch", "split", "total_loss", "ce_loss", "kd_loss", "accuracy"]

            # Add method-specific loss columns
            method_headers = {
                "FitNet": ["hint_loss"],
                "CRD": ["contrastive_loss"],
                "DKD": ["tckd_loss", "nckd_loss"],
            }

            if method in method_headers:
                headers.extend(method_headers[method])

        self.headers = headers
        self._write_headers()

    def _write_headers(self):
        """Write CSV headers."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log_epoch(self, epoch: int, phase: str, losses: dict, accuracy: float):
        """
        Log losses and metrics for an epoch.

        For DisDKD, 'phase' should be 'train' or 'val', and losses dict should contain
        'disdkd_phase' key with value 1, 2, or 3.
        """
        if self.method == "DisDKD":
            self._log_disdkd_epoch(epoch, phase, losses, accuracy)
        else:
            self._log_standard_epoch(epoch, phase, losses, accuracy)

    def _log_disdkd_epoch(self, epoch: int, split: str, losses: dict, accuracy: float):
        """Log DisDKD-specific metrics."""
        disdkd_phase = losses.get("disdkd_phase", 0)

        row = [
            epoch,
            disdkd_phase,
            split,
            losses.get("disc", ""),  # Phase 1
            losses.get("disc_acc", ""),  # Phase 1
            losses.get("adversarial", ""),  # Phase 2
            losses.get("fool_rate", ""),  # Phase 2
            losses.get("ce", ""),  # Phase 3
            losses.get("dkd", ""),  # Phase 3
            losses.get("total", ""),
            accuracy if accuracy > 0 else "",
        ]

        self._write_row(row)

    def _log_standard_epoch(
        self, epoch: int, phase: str, losses: dict, accuracy: float
    ):
        """Log standard method metrics."""
        row = [
            epoch,
            phase,
            losses.get("total", 0),
            losses.get("ce", 0),
            losses.get("kd", 0),
            accuracy,
        ]

        # Add method-specific losses
        method_losses = {
            "FitNet": ["hint"],
            "CRD": ["contrastive"],
            "DKD": ["tckd", "nckd"],
        }

        if self.method in method_losses:
            for loss_key in method_losses[self.method]:
                row.append(losses.get(loss_key, 0))

        self._write_row(row)

    def _write_row(self, row):
        """Append a row to the CSV file."""
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
