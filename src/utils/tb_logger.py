# src/utils/tb_logger.py
from pathlib import Path
from typing import Dict, Optional, Union
from torch.utils.tensorboard import SummaryWriter
import datetime as _dt

class TBLogger:
    """
    Thin wrapper around SummaryWriter.
    Usage:
        tb = TBLogger(root_logdir)
        tb.log_scalar("train/loss", 0.12, step=10)
        tb.log_scalars("train/losses", {"policy": 0.2, "value": 0.1}, step=10)
        tb.close()
    """
    def __init__(self, root_logdir: Union[str, Path], run_name: Optional[str] = None):
        root = Path(root_logdir)
        root.mkdir(parents=True, exist_ok=True)
        if run_name is None:
            stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = f"run-{stamp}"
        self.logdir = root / run_name
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.logdir))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, float(value), step)

    def log_scalars(self, tag: str, values: Dict[str, float], step: int) -> None:
        self.writer.add_scalars(tag, {k: float(v) for k, v in values.items()}, step)

    def log_hist(self, tag: str, values, step: int, bins: str = "tensorflow") -> None:
        # values: np.ndarray or torch.Tensor
        self.writer.add_histogram(tag, values, step, bins=bins)

    def add_text(self, tag: str, text: str, step: int) -> None:
        self.writer.add_text(tag, text, step)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()
