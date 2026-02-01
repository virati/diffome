"""Abstract base classes and interfaces for diffome components."""
from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


class PointCloud(ABC):
    """Abstract interface for point cloud data."""

    @abstractmethod
    def get_points(self) -> np.ndarray:
        """Return the point cloud as a numpy array."""
        pass

    @abstractmethod
    def subsample(self, factor: int):
        """Subsample the point cloud."""
        pass


class StreamlineLoader(ABC):
    """Abstract interface for loading streamline data."""

    @abstractmethod
    def load(self, filename: str, reference: Optional[Any] = None) -> Any:
        """Load streamlines from a file."""
        pass


class Renderer(ABC):
    """Abstract base class for rendering visualizations."""

    @abstractmethod
    def render(self, data: Any) -> None:
        """Render the given data."""
        pass

    @abstractmethod
    def display(self, data: Any) -> None:
        """Display the rendered data."""
        pass
