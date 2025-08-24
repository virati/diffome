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
        # calculate barcode on connectome
        active_streamlines = self.input_connectome.streamlines.streamlines
        active_streamlines = np.concatenate(active_streamlines)
        print(f"Rips on {active_streamlines.shape} streamlines...")

        # Create a RipsComplex from the active streamlines
        rips_complex = gd.RipsComplex(points=active_streamlines)

        # Generate the simplex tree
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

        # Compute the persistence
        persistence = simplex_tree.persistence()
        gd.plot_persistence_diagram(persistence)

        self.barcode = persistence

        return self

    def plot_barcode(self):
        pass
