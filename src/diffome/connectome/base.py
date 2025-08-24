import logging
import diffome.viz
from dipy.io.streamline import load_tractogram


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
        self.streamlines = self._streamlines[::factor]
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
