"""Configuration for unsupervised trajectory anomaly detection."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AnomalyConfig:
    """Settings shared by timestep- and window-level anomaly detectors."""

    pca_prefix: str = "z_"
    cluster_col: str = "final_cluster_label"
    trajectory_col: str | None = None
    year_col: str = "year"
    week_col: str = "week"
    window_size: int = 12
    window_stride: int = 1
    contamination: float = 0.05
    warning_quantile: float = 0.95
    high_quantile: float = 0.99
    n_estimators: int = 200
    random_state: int = 42
    min_cluster_samples: int = 40

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
        if self.window_stride < 1:
            raise ValueError("window_stride must be at least 1")
        if not 0.0 < self.contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")
        if not 0.0 < self.warning_quantile < self.high_quantile < 1.0:
            raise ValueError("Require 0 < warning_quantile < high_quantile < 1")
        if self.n_estimators < 10:
            raise ValueError("n_estimators must be at least 10")

    def to_dict(self) -> dict:
        """Return JSON-serializable configuration metadata."""
        return asdict(self)
