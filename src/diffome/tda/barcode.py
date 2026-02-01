from typing import Optional, Any
from diffome.connectome.base import Connectome
from diffome.tda.persistence import PersistenceComputer, PersistenceDiagramPlotter


class TDAAnalysis:
    """Base class for topological data analysis."""

    def __init__(self, input_connectome: Connectome):
        """Initialize TDA analysis with a connectome.

        Args:
            input_connectome: Connectome object to analyze
        """
        self.input_connectome = input_connectome
        self.result = None


class BarCode(TDAAnalysis):
    """Barcode analysis using persistent homology."""

    def __init__(self, input_connectome: Connectome):
        """Initialize barcode analysis.

        Args:
            input_connectome: Connectome object to analyze
        """
        super().__init__(input_connectome)
        self.barcode = None
        self.computer = PersistenceComputer()
        self.plotter = PersistenceDiagramPlotter()

    def calculate(self, params: Optional[dict] = None, do_plot: bool = True):
        """Calculate persistence barcode for the connectome.

        Args:
            params: Optional parameters for computation
            do_plot: Whether to plot the persistence diagram

        Returns:
            Self for method chaining
        """
        # Extract point cloud from connectome
        points = self.input_connectome.get_points()

        if points is None:
            raise ValueError("No points available. Ensure connectome is properly loaded and subsampled.")

        # Compute persistence
        max_dimension = 2
        if params and "max_dimension" in params:
            max_dimension = params["max_dimension"]

        self.barcode = self.computer.compute_rips_persistence(points, max_dimension)

        # Plot if requested
        if do_plot:
            self.plot_barcode()

        return self

    def plot_barcode(self):
        """Plot the computed persistence diagram."""
        if self.barcode is None:
            raise ValueError("No barcode computed. Call calculate() first.")
        self.plotter.plot(self.barcode)
