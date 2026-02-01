import gudhi as gd
from diffome.connectome.base import Connectome
import numpy as np


class TDAAnalysis:
    def __init__(self, input_connectome: Connectome):
        self.input_connectome = input_connectome


class BarCode(TDAAnalysis):
    def __init__(self, input_connectome: Connectome):
        super().__init__(input_connectome)

    def calculate(self, params: dict = None, do_plot=True) -> None:
        """
        Calculate barcode on connectome using optimized Rips complex construction.
        
        Args:
            params: Optional dictionary with parameters:
                - max_edge_length (float): Maximum edge length for sparse Rips complex.
                  Smaller values lead to faster computation. Default: 5.0
                - use_edge_collapse (bool): Whether to use edge collapse optimization
                  for faster persistence computation. Default: True
            do_plot: Whether to plot the persistence diagram
            
        Returns:
            self: Returns the BarCode instance for method chaining
        """
        # Parse parameters with defaults
        if params is None:
            params = {}
        max_edge_length = params.get('max_edge_length', 5.0)
        use_edge_collapse = params.get('use_edge_collapse', True)
        
        # calculate barcode on connectome
        active_streamlines = self.input_connectome.streamlines.streamlines
        active_streamlines = np.concatenate(active_streamlines)
        print(f"Rips on {active_streamlines.shape} streamlines...")

        # Create a sparse RipsComplex from the active streamlines with max_edge_length
        # This implements the "flood complex" approach for faster computation by
        # limiting edge creation to nearby points only (sparse Rips)
        rips_complex = gd.RipsComplex(points=active_streamlines, max_edge_length=max_edge_length)

        # Generate the simplex tree
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

        # Apply edge collapse optimization to reduce redundant edges
        # This further speeds up persistence computation while preserving topology
        if use_edge_collapse:
            simplex_tree.collapse_edges()

        # Compute the persistence
        persistence = simplex_tree.persistence()
        if do_plot:
            gd.plot_persistence_diagram(persistence)

        self.barcode = persistence

        return self

    def plot_barcode(self):
        pass
