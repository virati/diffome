"""TDA computation components for persistence analysis."""
import gudhi as gd
import numpy as np
from typing import Any, Optional


class PersistenceComputer:
    """Compute persistence diagrams using topological data analysis."""

    @staticmethod
    def compute_rips_persistence(
        points: np.ndarray, max_dimension: int = 2
    ) -> list:
        """Compute Rips complex and persistence diagram.

        Args:
            points: Point cloud data as numpy array
            max_dimension: Maximum dimension for simplex tree

        Returns:
            Persistence diagram as list of tuples
        """
        print(f"Computing Rips complex on {points.shape} points...")

        # Create a RipsComplex from the points
        rips_complex = gd.RipsComplex(points=points)

        # Generate the simplex tree
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)

        # Compute and return the persistence
        return simplex_tree.persistence()


class PersistenceDiagramPlotter:
    """Visualize persistence diagrams."""

    @staticmethod
    def plot(persistence: list, **kwargs) -> None:
        """Plot a persistence diagram.

        Args:
            persistence: Persistence diagram data
            **kwargs: Additional plotting parameters
        """
        gd.plot_persistence_diagram(persistence)
