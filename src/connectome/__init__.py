"""
Connectome: A comprehensive toolkit for topological analysis of brain white matter tractography.

This package provides tools for:
- Loading and processing tractography data (.trk files)
- Extracting spatial point clouds from streamlines
- Computing persistent homology for topological analysis
- Calculating distance metrics between connectomes
- Generating publication-ready visualizations
- Performing robustness analysis through sampling strategies
"""

__version__ = "1.0.0"
__author__ = "Connectome Analysis Team"

# Import main classes and functions for easy access
from .tractography import TractogramLoader, find_tractography_file
from .topology import analyze_tractography_topology
from .visualization import TractographyVisualizer
from .connectome_statistics import TractographyStatistics
from .tda_metrics import PersistenceDiagramMetrics
from .sampling import StreamlineSampler

__all__ = [
    "TractogramLoader",
    "find_tractography_file", 
    "analyze_tractography_topology",
    "TractographyVisualizer",
    "TractographyStatistics",
    "PersistenceDiagramMetrics",
    "StreamlineSampler",
]
