import gudhi as gd
from diffome.connectome.base import Connectome
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TDAAnalysis:
    def __init__(self, input_connectome: Connectome):
        self.input_connectome = input_connectome


class BarCode(TDAAnalysis):
    def __init__(self, input_connectome: Connectome):
        super().__init__(input_connectome)

    def calculate(
        self,
        params: dict = None,
        do_plot=True,
        ignore_streamlines=None,
        downsample_points=None,
    ) -> None:
        if ignore_streamlines is None:
            ignore_streamlines = []
        # calculate barcode on connectome

        active_streamlines = self.input_connectome.streamlines.streamlines
        if len(ignore_streamlines) > 0:
            keep_idx = [
                sl
                for sl in range(len(active_streamlines))
                if sl not in ignore_streamlines
            ]

            active_streamlines = active_streamlines[keep_idx]
        active_points = np.concatenate(active_streamlines)
        if downsample_points is not None:
            active_points = active_points[::downsample_points]
        logging.info(f"Rips on {active_points.shape} points...")

        # Create a RipsComplex from the active streamlines
        rips_complex = gd.RipsComplex(points=active_points)

        # Generate the simplex tree
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

        # Compute the persistence
        persistence = simplex_tree.persistence()
        if do_plot:
            gd.plot_persistence_diagram(persistence)

        self.barcode = persistence

        return self

    def plot_barcode(self):
        pass
