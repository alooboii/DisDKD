import csv
from pathlib import Path


class LossTracker:
    def __init__(self, log_path: str, method: str):
        self.log_path = Path(log_path)
        self.method = method

        # Standard headers common to all methods (Phase 2 uses these)
        self.headers = ["epoch", "phase", "total", "ce", "kd", "accuracy"]

        method_headers = {
            "FitNet": ["hint"],
            "CRD": ["contrastive"],
            "DKD": ["tckd", "nckd"],
            "DisDKD": [
                "dkd",  # Phase 2
                "discriminator",  # Phase 1
                "generator",  # Phase 1 (Added)
                "adversarial",  # Phase 1
                "disc_acc",  # Phase 1 (Renamed from disc_accuracy to match Trainer)
                "fool_rate",  # Phase 1
            ],
        }
        if method in method_headers:
            self.headers.extend(method_headers[method])

        self.headers.append("lr")

        self._write_headers()

    def _write_headers(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(self.headers)

    @staticmethod
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return 0.0

    def log_epoch(
        self, epoch: int, phase: str, losses: dict, accuracy: float, lr: float = 0.0
    ):
        row = [
            epoch,
            phase,
            self._to_float(losses.get("total", 0.0)),
            self._to_float(losses.get("ce", 0.0)),
            self._to_float(losses.get("kd", 0.0)),
            self._to_float(accuracy),
        ]

        method_losses = {
            "FitNet": ["hint"],
            "CRD": ["contrastive"],
            "DKD": ["tckd", "nckd"],
            "DisDKD": [
                "dkd",
                "discriminator",
                "generator",
                "adversarial",
                "disc_acc",
                "fool_rate",
            ],
        }

        # This loop handles the sparse updates nicely.
        # In Phase 1: 'dkd' will be 0.0
        # In Phase 2: 'discriminator' etc. will be 0.0
        for k in method_losses.get(self.method, []):
            row.append(self._to_float(losses.get(k, 0.0)))

        row.append(self._to_float(lr))

        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
