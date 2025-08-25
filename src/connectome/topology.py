#!/usr/bin/env python3
"""
Topology Module
==============

Topological Data Analysis (TDA) for tractography data.
Implements point cloud extraction and persistent homology computation.
"""

import numpy as np

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError as e:
    HAS_RIPSER = False
    _ripser_error = str(e)


class PointCloudExtractor:
    """
    Extracts and preprocesses point clouds from streamlines for TDA.
    
    Provides various downsampling strategies to make TDA computationally feasible
    while preserving important spatial structure.
    """
    
    def __init__(self, random_seed=42):
        """
        Initialize extractor with random seed for reproducibility.
        
        Args:
            random_seed (int): Random seed for sampling operations
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def extract_all_points(self, streamlines):
        """
        Extract all vertex coordinates from streamlines.
        
        Args:
            streamlines (list): List of streamline arrays (each Nx3)
            
        Returns:
            np.ndarray: Mx3 array of all coordinates
        """
        if not streamlines:
            raise ValueError("No streamlines provided")
            
        # Concatenate all streamline points
        all_points = np.concatenate([s for s in streamlines if len(s) > 0])
        
        if len(all_points) == 0:
            raise ValueError("No points found in streamlines")
            
        return all_points
    
    def voxel_downsample(self, points, voxel_size):
        """
        Downsample points using voxel grid.
        
        This quantizes the space into voxels and keeps one point per voxel,
        significantly reducing point density while preserving spatial structure.
        
        Args:
            points (np.ndarray): Nx3 array of coordinates
            voxel_size (float): Voxel size in mm
            
        Returns:
            np.ndarray: Downsampled points
        """
        # Quantize points to voxel grid
        quantized = np.floor(points / voxel_size).astype(np.int64)
        
        # Find unique voxels and their indices
        _, unique_indices = np.unique(quantized, axis=0, return_index=True)
        
        return points[unique_indices]
    
    def random_subsample(self, points, max_points):
        """
        Randomly subsample points to a maximum number.
        
        Args:
            points (np.ndarray): Nx3 array of coordinates  
            max_points (int): Maximum number of points to keep
            
        Returns:
            np.ndarray: Subsampled points
        """
        if len(points) <= max_points:
            return points
            
        indices = np.random.choice(len(points), max_points, replace=False)
        return points[indices]
    
    def create_point_cloud(self, streamlines, max_points=1500, voxel_size=2.0):
        """
        Create a downsampled point cloud suitable for TDA.
        
        Applies voxel downsampling followed by random subsampling to achieve
        a computationally tractable point cloud that preserves spatial structure.
        
        Args:
            streamlines (list): List of streamline arrays
            max_points (int): Maximum number of points in final cloud
            voxel_size (float): Voxel size for grid downsampling (mm)
            
        Returns:
            np.ndarray: Final point cloud (Mx3)
        """
        print(f"Creating point cloud (max: {max_points}, voxel: {voxel_size}mm)")
        
        # Extract all points
        all_points = self.extract_all_points(streamlines)
        print(f"  Initial points: {len(all_points):,}")
        
        # Voxel downsampling
        voxel_points = self.voxel_downsample(all_points, voxel_size)
        print(f"  After voxel downsampling: {len(voxel_points):,}")
        
        # Random subsampling if needed
        final_points = self.random_subsample(voxel_points, max_points)
        print(f"  Final point cloud: {len(final_points):,}")
        
        return final_points


class PersistentHomologyAnalyzer:
    """
    Computes persistent homology on point clouds using Vietoris-Rips complexes.
    
    Provides analysis of topological features (connected components, loops, voids)
    and their persistence across different scales.
    """
    
    def __init__(self):
        """Initialize the TDA analyzer."""
        if not HAS_RIPSER:
            raise ImportError(f"ripser package required for TDA: {_ripser_error}")
    
    def compute_persistence(self, point_cloud, max_dimension=1, max_edge_length=None):
        """
        Compute persistent homology of a point cloud.
        
        Uses Vietoris-Rips complexes with Euclidean distances to build
        a filtration and compute persistence diagrams.
        
        Args:
            point_cloud (np.ndarray): Nx3 array of coordinates
            max_dimension (int): Maximum homology dimension to compute
            max_edge_length (float, optional): Maximum edge length for VR complex
            
        Returns:
            list: Persistence diagrams for H0, H1, ..., H(max_dimension)
        """
        print(f"Computing persistent homology (dim ≤ {max_dimension})")
        print(f"  Point cloud size: {len(point_cloud)}")
        
        # Set up ripser parameters
        ripser_params = {
            'maxdim': max_dimension,
            'distance_matrix': False  # Use point cloud directly
        }
        
        if max_edge_length is not None:
            ripser_params['thresh'] = max_edge_length
            print(f"  Max edge length: {max_edge_length}")
        
        # Compute persistence
        result = ripser(point_cloud, **ripser_params)
        diagrams = result['dgms']
        
        # Print summary
        for dim, diagram in enumerate(diagrams):
            n_features = len(diagram)
            if n_features > 0:
                finite_mask = np.isfinite(diagram).all(axis=1)
                n_finite = np.sum(finite_mask)
                n_infinite = n_features - n_finite
                
                if n_finite > 0:
                    lifetimes = diagram[finite_mask, 1] - diagram[finite_mask, 0]
                    max_life = np.max(lifetimes)
                    total_life = np.sum(lifetimes)
                    print(f"  H{dim}: {n_features} features ({n_finite} finite, {n_infinite} infinite)")
                    print(f"       Max persistence: {max_life:.3f}")
                    print(f"       Total persistence: {total_life:.3f}")
                else:
                    print(f"  H{dim}: {n_features} features (all infinite)")
            else:
                print(f"  H{dim}: 0 features")
        
        return diagrams
    
    def analyze_features(self, diagrams):
        """
        Extract quantitative measures from persistence diagrams.
        
        Args:
            diagrams (list): Persistence diagrams from compute_persistence
            
        Returns:
            dict: Dictionary of topological features and statistics
        """
        analysis = {}
        
        for dim, diagram in enumerate(diagrams):
            dim_name = f'H{dim}'
            
            if len(diagram) == 0:
                analysis[dim_name] = {
                    'n_features': 0,
                    'n_finite': 0,
                    'n_infinite': 0,
                    'total_persistence': 0.0,
                    'max_persistence': 0.0,
                    'mean_persistence': 0.0
                }
                continue
            
            # Separate finite and infinite features
            finite_mask = np.isfinite(diagram).all(axis=1)
            finite_diagram = diagram[finite_mask]
            
            n_features = len(diagram)
            n_finite = len(finite_diagram)
            n_infinite = n_features - n_finite
            
            # Compute persistence statistics for finite features
            if n_finite > 0:
                lifetimes = finite_diagram[:, 1] - finite_diagram[:, 0]
                total_persistence = np.sum(lifetimes)
                max_persistence = np.max(lifetimes)
                mean_persistence = np.mean(lifetimes)
            else:
                total_persistence = 0.0
                max_persistence = 0.0
                mean_persistence = 0.0
            
            analysis[dim_name] = {
                'n_features': n_features,
                'n_finite': n_finite,
                'n_infinite': n_infinite,
                'total_persistence': total_persistence,
                'max_persistence': max_persistence,
                'mean_persistence': mean_persistence
            }
        
        return analysis
    
    def print_summary(self, analysis):
        """
        Print a human-readable summary of TDA results.
        
        Args:
            analysis (dict): Analysis results from analyze_features
        """
        print("\nTopological Analysis Summary:")
        print("=" * 40)
        
        for dim_name, stats in analysis.items():
            print(f"\n{dim_name} (dimension {dim_name[1]}):")
            print(f"  Features: {stats['n_features']} total")
            
            if stats['n_finite'] > 0:
                print(f"    Finite: {stats['n_finite']}")
                print(f"    Persistence: {stats['total_persistence']:.3f} total, {stats['max_persistence']:.3f} max")
                
            if stats['n_infinite'] > 0:
                print(f"    Infinite: {stats['n_infinite']}")
        
        # Interpretation
        print(f"\nInterpretation:")
        print(f"  H0: Connected components in the white matter")
        print(f"  H1: Loops or holes in the fiber structure")
        print(f"  Higher persistence indicates more significant features")


def analyze_tractography_topology(streamlines, max_points=1500, voxel_size=2.0, 
                                max_dimension=1, random_seed=42):
    """
    Complete TDA pipeline for tractography data.
    
    Args:
        streamlines (list): List of streamline arrays
        max_points (int): Maximum points for TDA computation
        voxel_size (float): Voxel size for downsampling
        max_dimension (int): Maximum homology dimension
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (point_cloud, diagrams, analysis)
    """
    # Extract point cloud
    extractor = PointCloudExtractor(random_seed=random_seed)
    point_cloud = extractor.create_point_cloud(
        streamlines, max_points=max_points, voxel_size=voxel_size
    )
    
    # Compute persistent homology
    analyzer = PersistentHomologyAnalyzer()
    diagrams = analyzer.compute_persistence(point_cloud, max_dimension=max_dimension)
    
    # Analyze results
    analysis = analyzer.analyze_features(diagrams)
    analyzer.print_summary(analysis)
    
    return point_cloud, diagrams, analysis
