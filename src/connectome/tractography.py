#!/usr/bin/env python3
"""
Tractography Module
==================

Core functionality for loading and analyzing white matter tractography data.
Handles TRK file I/O and basic streamline operations.
"""

import os
import numpy as np
from dipy.io.streamline import load_tractogram


class TractogramLoader:
    """
    Handles loading and basic processing of tractography files.
    
    Supports .trk format via DIPY. Provides coordinate space management
    and basic streamline statistics.
    """
    
    def __init__(self, file_path):
        """
        Initialize loader with TRK file path.
        
        Args:
            file_path (str): Path to .trk tractography file
        """
        self.file_path = file_path
        self.sft = None
        self.streamlines = None
        self._stats = None
        
    def load(self):
        """
        Load the tractography file.
        
        Returns:
            self: For method chaining
            
        Raises:
            FileNotFoundError: If TRK file doesn't exist
            ValueError: If file is corrupted or empty
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Tractography file not found: {self.file_path}")
            
        # Load using DIPY
        self.sft = load_tractogram(self.file_path, 'same')
        
        if len(self.sft) == 0:
            raise ValueError("Empty tractography file")
            
        # Convert to world coordinates (RAS mm) for consistency
        self.sft.to_rasmm()
        self.streamlines = list(self.sft.streamlines)
        
        return self
        
    def get_statistics(self):
        """
        Compute basic streamline statistics.
        
        Returns:
            dict: Statistics including counts, lengths, and spatial extent
        """
        if self.streamlines is None:
            raise ValueError("Must load tractography first")
            
        if self._stats is not None:
            return self._stats
            
        # Streamline counts and lengths
        n_streamlines = len(self.streamlines)
        lengths = [len(s) for s in self.streamlines]
        
        # Spatial extent
        all_points = np.concatenate([s for s in self.streamlines])
        
        self._stats = {
            'n_streamlines': n_streamlines,
            'n_points_total': len(all_points),
            'length_mean': np.mean(lengths),
            'length_std': np.std(lengths),
            'length_min': min(lengths),
            'length_max': max(lengths),
            'spatial_extent': {
                'x_range': (all_points[:, 0].min(), all_points[:, 0].max()),
                'y_range': (all_points[:, 1].min(), all_points[:, 1].max()),
                'z_range': (all_points[:, 2].min(), all_points[:, 2].max())
            }
        }
        
        return self._stats
        
    def print_summary(self):
        """Print a human-readable summary of the tractography."""
        stats = self.get_statistics()
        
        print(f"Tractography Summary:")
        print(f"  File: {os.path.basename(self.file_path)}")
        print(f"  Streamlines: {stats['n_streamlines']:,}")
        print(f"  Total points: {stats['n_points_total']:,}")
        print(f"  Length: {stats['length_mean']:.1f} ± {stats['length_std']:.1f} points")
        print(f"  Range: {stats['length_min']} - {stats['length_max']} points")
        
        extent = stats['spatial_extent']
        print(f"  Spatial extent (mm):")
        print(f"    X: {extent['x_range'][0]:.1f} to {extent['x_range'][1]:.1f}")
        print(f"    Y: {extent['y_range'][0]:.1f} to {extent['y_range'][1]:.1f}")  
        print(f"    Z: {extent['z_range'][0]:.1f} to {extent['z_range'][1]:.1f}")


def find_tractography_file(search_paths=None):
    """
    Find available tractography files in the current directory.
    
    Args:
        search_paths (list, optional): List of file paths to check.
                                     Defaults to common names.
    
    Returns:
        str or None: Path to first found TRK file, or None if none found
    """
    if search_paths is None:
        search_paths = [
            'data/NeurReps Top Right.trk',
            'data/Petersen PD Top Left.trk',
            'data/Peterson PD Top Left.trk', 
            'data/Peterson PD Top Right.trk',
            'data/*.trk',  # Look in data folder first
            '*.trk'  # Fallback for files in root directory
        ]
    
    for path in search_paths:
        if os.path.exists(path):
            return path
            
    return None
