import logging
import diffome.viz
from dipy.io.streamline import load_tractogram
import numpy as np


class Connectome:
    def __init__(self, input_streamlines=None, ref=None):
        if input_streamlines is None:
            # assume synthetic here...
            input_streamlines = self.generate_synthetic_streamlines()
        else:
            input_streamlines = load_tractogram(
                input_streamlines,
                ref,
            )
        # print(input_streamlines)
        self._streamlines = input_streamlines
        self.streamlines = None  # active streamlines

    def subsample(self, factor: int = 10):
        if factor <= 1:
            logging.warning("Subsample factor should be greater than 1.")
            return self
        n_total_streamlines = len(self._streamlines)
        n_keep_streamlines = n_total_streamlines // factor
        random_indices = np.random.choice(
            n_total_streamlines, n_keep_streamlines, replace=False
        )
        self.streamlines = self._streamlines[random_indices]
        return self

    def clip_streamlines(self, n_clip: int = 100):
        self.streamlines = self._streamlines[:n_clip]
        return self

    def generate_synthetic_streamlines(self):
        # Placeholder for synthetic streamline generation logic
        return []

    def render(self):
        if self.streamlines is None:
            logging.warning("No streamlines available for rendering.")
            return
        renderer = diffome.viz.base_renderer()
        renderer.display(self._streamlines)
