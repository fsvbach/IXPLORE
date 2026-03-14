import logging
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Package-level logger
logger: logging.Logger = logging.getLogger("ixplore")
logger.setLevel(logging.INFO)


@dataclass
class FitLogger:
    """Records metrics each time iterate() is called."""
    mae: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)

    def log(self, mae: float, accuracy: float) -> None:
        self.mae.append(mae)
        self.accuracy.append(accuracy)

    def clear(self) -> None:
        self.mae.clear()
        self.accuracy.clear()

