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
    boundary_fraction: list[float] = field(default_factory=list)

    def log(self, mae: float, accuracy: float, boundary_fraction: float) -> None:
        self.mae.append(mae)
        self.accuracy.append(accuracy)
        self.boundary_fraction.append(boundary_fraction)

    def clear(self) -> None:
        self.mae.clear()
        self.accuracy.clear()
        self.boundary_fraction.clear()

