import logging

# import diffome.viz
from dipy.io.streamline import load_tractogram
import numpy as np

from dipy.tracking.streamline import length, transform_streamlines
from dipy.viz import actor, window


class Connectome:
    def __init__(self, input_streamlines=None, ref=None, bbox_valid_check=True):
        if input_streamlines is None:
            # assume synthetic here...
            input_streamlines = self.generate_synthetic_streamlines()
        else:
            input_streamlines = load_tractogram(
                input_streamlines, ref, bbox_valid_check=bbox_valid_check
            )
        # print(input_streamlines)
        self._streamlines = input_streamlines
        self.streamlines = self._streamlines  # active streamlines

    def subsample(self, factor: int = 10):
        if factor < 1:
            logging.warning("Subsample factor should be >= 1.")
            return self
        if factor == 1:
            self.streamlines = self._streamlines
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

        logging.info(f"Clipped to {n_clip} streamlines.")
        return self

    def generate_synthetic_streamlines(self):
        # Placeholder for synthetic streamline generation logic
        return []

    def render(self):
        if self.streamlines is None:
            logging.warning("No streamlines available for rendering.")
            return
        scene = window.Scene()
        stream_actor = actor.line(self.streamlines.streamlines)

        # scene.set_camera(
        #    position=(-176.42, 118.52, 128.20),
        #    focal_point=(113.30, 128.31, 76.56),
        #    view_up=(0.18, 0.00, 0.98),
        # )

        scene.add(stream_actor)

        # Uncomment the line below to show to display the window
        window.show(scene, size=(600, 600), reset_camera=False)
