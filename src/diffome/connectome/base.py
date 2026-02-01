import logging
from typing import Optional, Any
from diffome.connectome.data_loader import (
    TractogramLoader,
    SyntheticStreamlineGenerator,
    PointCloudExtractor,
)
from diffome.core.interfaces import PointCloud


class Connectome(PointCloud):
    """Main class for managing connectome streamline data.

    This class provides methods for loading, subsampling, and accessing
    streamline data from connectome files.
    """

    def __init__(self, input_streamlines=None, ref=None):
        """Initialize Connectome.

        Args:
            input_streamlines: Path to streamline file or None for synthetic data
            ref: Reference anatomy for loading tractogram
        """
        if input_streamlines is None:
            # Generate synthetic streamlines
            self._streamlines = SyntheticStreamlineGenerator.generate()
        else:
            # Load streamlines from file
            loader = TractogramLoader()
            self._streamlines = loader.load(input_streamlines, ref)

        self.streamlines = None  # active streamlines after subsampling

    def subsample(self, factor: int = 10):
        """Subsample the streamlines by the given factor.

        Args:
            factor: Subsampling factor (must be > 1)

        Returns:
            Self for method chaining
        """
        if factor <= 1:
            logging.warning("Subsample factor should be greater than 1.")
            return self
        self.streamlines = self._streamlines[::factor]
        return self

    def get_points(self):
        """Extract point cloud from streamlines.

        Returns:
            Numpy array of points from streamlines
        """
        if self.streamlines is None:
            logging.warning("No subsampled streamlines available. Call subsample() first.")
            return None
        return PointCloudExtractor.extract_from_streamlines(self.streamlines)

    def render(self, renderer: Optional[Any] = None):
        """Render the connectome using the provided renderer.

        Args:
            renderer: Renderer instance to use for visualization
        """
        if renderer is None:
            from diffome.viz import create_streamline_renderer
            renderer = create_streamline_renderer()

        if self.streamlines is None:
            logging.warning("No streamlines available for rendering.")
            return

        renderer.render(self._streamlines)
        renderer.display(self._streamlines)
