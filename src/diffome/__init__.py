"""Diffome - Python library to compare connectomes in various ways."""
from diffome.connectome import Connectome
from diffome.tda import BarCode, TDAAnalysis
from diffome.viz import create_streamline_renderer

__version__ = "0.1.0"

__all__ = [
    "Connectome",
    "BarCode",
    "TDAAnalysis",
    "create_streamline_renderer",
]