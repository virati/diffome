#!/usr/bin/env python3
"""
Streamline Sampling Module
=========================

Advanced sampling strategies for studying TDA robustness and sampling effects.
Implements multiple sampling approaches to understand how topology changes
with different data subsets.
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')


class StreamlineSampler:
    """
    Implements various streamline sampling strategies for TDA robustness studies.
    
    Supports uniform sampling, topology-preserving sampling, and adaptive sampling
    methods to understand how different data subsets affect topological features.
    """
    
    def __init__(self, streamlines, random_seed=None):
        """
        Initialize sampler with streamlines.
        
        Args:
            streamlines (list): List of streamline arrays (each Nx3)
            random_seed (int, optional): Random seed for reproducibility. If None, uses random seed.
        """
        self.streamlines = streamlines
        self.n_streamlines = len(streamlines)
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Precompute streamline properties for efficient sampling
        self._compute_streamline_properties()
        
    def _compute_streamline_properties(self):
        """Precompute properties needed for various sampling strategies."""
        print("Computing streamline properties for sampling...")
        
        # Length statistics
        self.lengths = np.array([len(s) for s in self.streamlines])
        
        # Spatial statistics (endpoints, centroids)
        self.start_points = np.array([s[0] for s in self.streamlines])
        self.end_points = np.array([s[-1] for s in self.streamlines])
        self.centroids = np.array([np.mean(s, axis=0) for s in self.streamlines])
        
        # Spatial extent for each streamline
        self.spatial_extents = np.array([
            np.linalg.norm(s.max(axis=0) - s.min(axis=0)) 
            for s in self.streamlines
        ])
        
        print(f"  Computed properties for {self.n_streamlines} streamlines")
        
    def uniform_sampling(self, k=10):
        """
        Uniform random sampling of streamlines.
        
        Args:
            k (int): Sampling factor (take 1/k of streamlines)
            
        Returns:
            tuple: (sampled_streamlines, indices, metadata)
        """
        n_sample = max(1, self.n_streamlines // k)
        indices = np.random.choice(self.n_streamlines, n_sample, replace=False)
        indices = np.sort(indices)
        
        sampled_streamlines = [self.streamlines[i] for i in indices]
        
        metadata = {
            'strategy': 'uniform',
            'k': k,
            'n_original': self.n_streamlines,
            'n_sampled': n_sample,
            'sampling_rate': n_sample / self.n_streamlines
        }
        
        return sampled_streamlines, indices, metadata
    
    def length_stratified_sampling(self, k=10):
        """
        Sample streamlines stratified by length to preserve length distribution.
        
        Args:
            k (int): Sampling factor
            
        Returns:
            tuple: (sampled_streamlines, indices, metadata)
        """
        n_sample = max(1, self.n_streamlines // k)
        
        # Create length-based strata
        length_percentiles = np.percentile(self.lengths, [0, 25, 50, 75, 100])
        strata_indices = []
        
        for i in range(len(length_percentiles) - 1):
            mask = (self.lengths >= length_percentiles[i]) & (self.lengths < length_percentiles[i+1])
            if i == len(length_percentiles) - 2:  # Include max in last stratum
                mask = (self.lengths >= length_percentiles[i]) & (self.lengths <= length_percentiles[i+1])
            strata_indices.append(np.where(mask)[0])
        
        # Sample proportionally from each stratum
        indices = []
        for stratum in strata_indices:
            if len(stratum) > 0:
                n_from_stratum = max(1, len(stratum) * n_sample // self.n_streamlines)
                if len(stratum) >= n_from_stratum:
                    sampled = np.random.choice(stratum, n_from_stratum, replace=False)
                    indices.extend(sampled)
        
        indices = np.array(indices)
        indices = np.sort(indices)
        sampled_streamlines = [self.streamlines[i] for i in indices]
        
        metadata = {
            'strategy': 'length_stratified',
            'k': k,
            'n_original': self.n_streamlines,
            'n_sampled': len(indices),
            'sampling_rate': len(indices) / self.n_streamlines,
            'length_strata': length_percentiles
        }
        
        return sampled_streamlines, indices, metadata
    
    def spatial_clustering_sampling(self, k=10, n_clusters=None):
        """
        Sample streamlines using spatial clustering to preserve spatial topology.
        
        Args:
            k (int): Sampling factor
            n_clusters (int, optional): Number of clusters. Auto-determined if None.
            
        Returns:
            tuple: (sampled_streamlines, indices, metadata)
        """
        n_sample = max(1, self.n_streamlines // k)
        
        # Determine number of clusters
        if n_clusters is None:
            n_clusters = min(n_sample, max(10, n_sample // 2))
        
        # Cluster based on streamline centroids
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(self.centroids)
        
        # Sample from each cluster proportionally
        indices = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) > 0:
                # Sample proportionally from this cluster
                n_from_cluster = max(1, len(cluster_indices) * n_sample // self.n_streamlines)
                n_from_cluster = min(n_from_cluster, len(cluster_indices))
                
                sampled = np.random.choice(cluster_indices, n_from_cluster, replace=False)
                indices.extend(sampled)
        
        indices = np.array(indices)
        indices = np.sort(indices)
        sampled_streamlines = [self.streamlines[i] for i in indices]
        
        metadata = {
            'strategy': 'spatial_clustering',
            'k': k,
            'n_original': self.n_streamlines,
            'n_sampled': len(indices),
            'sampling_rate': len(indices) / self.n_streamlines,
            'n_clusters': n_clusters
        }
        
        return sampled_streamlines, indices, metadata
    
    def density_based_sampling(self, k=10, density_radius=5.0):
        """
        Sample streamlines based on local density to preserve topological structure.
        
        Args:
            k (int): Sampling factor
            density_radius (float): Radius for density estimation (mm)
            
        Returns:
            tuple: (sampled_streamlines, indices, metadata)
        """
        n_sample = max(1, self.n_streamlines // k)
        
        # Compute local density for each streamline centroid
        distances = squareform(pdist(self.centroids))
        densities = np.sum(distances < density_radius, axis=1)
        
        # Create density-based sampling probabilities (inverse density)
        # Higher probability for streamlines in lower density regions
        inv_densities = 1.0 / (densities + 1)  # Add 1 to avoid division by zero
        probabilities = inv_densities / np.sum(inv_densities)
        
        # Sample based on inverse density probabilities
        indices = np.random.choice(
            self.n_streamlines, 
            size=n_sample, 
            replace=False, 
            p=probabilities
        )
        indices = np.sort(indices)
        sampled_streamlines = [self.streamlines[i] for i in indices]
        
        metadata = {
            'strategy': 'density_based',
            'k': k,
            'n_original': self.n_streamlines,
            'n_sampled': n_sample,
            'sampling_rate': n_sample / self.n_streamlines,
            'density_radius': density_radius,
            'mean_density': np.mean(densities)
        }
        
        return sampled_streamlines, indices, metadata
    
    def topology_preserving_sampling(self, k=10):
        """
        Advanced sampling that attempts to preserve topological structure.
        
        Combines spatial clustering and density-based sampling to maintain
        both global structure and local topology.
        
        Args:
            k (int): Sampling factor
            
        Returns:
            tuple: (sampled_streamlines, indices, metadata)
        """
        n_sample = max(1, self.n_streamlines // k)
        
        # Step 1: Spatial clustering to identify major bundles
        n_clusters = min(20, max(5, n_sample // 5))
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(self.centroids)
        
        # Step 2: Within each cluster, use density-based sampling
        indices = []
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
                
            # Target number from this cluster
            n_from_cluster = max(1, len(cluster_indices) * n_sample // self.n_streamlines)
            n_from_cluster = min(n_from_cluster, len(cluster_indices))
            
            if len(cluster_indices) <= n_from_cluster:
                # Take all streamlines from small clusters
                indices.extend(cluster_indices)
            else:
                # Density-based sampling within cluster
                cluster_centroids = self.centroids[cluster_indices]
                if len(cluster_centroids) > 1:
                    distances = squareform(pdist(cluster_centroids))
                    densities = np.sum(distances < 3.0, axis=1)  # Smaller radius for within-cluster
                    
                    # Prefer lower density streamlines (better coverage)
                    inv_densities = 1.0 / (densities + 1)
                    probabilities = inv_densities / np.sum(inv_densities)
                    
                    selected = np.random.choice(
                        len(cluster_indices),
                        size=n_from_cluster,
                        replace=False,
                        p=probabilities
                    )
                    indices.extend(cluster_indices[selected])
                else:
                    indices.extend(cluster_indices)
        
        indices = np.array(indices)
        indices = np.sort(indices)
        sampled_streamlines = [self.streamlines[i] for i in indices]
        
        metadata = {
            'strategy': 'topology_preserving',
            'k': k,
            'n_original': self.n_streamlines,
            'n_sampled': len(indices),
            'sampling_rate': len(indices) / self.n_streamlines,
            'n_clusters': n_clusters
        }
        
        return sampled_streamlines, indices, metadata
    
    def multi_strategy_sampling(self, k_values=[5, 10, 20], strategies=None):
        """
        Apply multiple sampling strategies and k values for comprehensive analysis.
        
        Args:
            k_values (list): List of k values to test
            strategies (list, optional): List of strategy names to use
            
        Returns:
            dict: Results for each strategy and k value combination
        """
        if strategies is None:
            strategies = [
                'uniform', 
                'length_stratified', 
                'spatial_clustering', 
                'density_based', 
                'topology_preserving'
            ]
        
        results = {}
        
        for strategy in strategies:
            print(f"\nApplying {strategy} sampling...")
            results[strategy] = {}
            
            for k in k_values:
                print(f"  k={k}...")
                
                if strategy == 'uniform':
                    sampled, indices, metadata = self.uniform_sampling(k)
                elif strategy == 'length_stratified':
                    sampled, indices, metadata = self.length_stratified_sampling(k)
                elif strategy == 'spatial_clustering':
                    sampled, indices, metadata = self.spatial_clustering_sampling(k)
                elif strategy == 'density_based':
                    sampled, indices, metadata = self.density_based_sampling(k)
                elif strategy == 'topology_preserving':
                    sampled, indices, metadata = self.topology_preserving_sampling(k)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                
                results[strategy][k] = {
                    'streamlines': sampled,
                    'indices': indices,
                    'metadata': metadata
                }
        
        return results
    
    def get_sampling_summary(self, results):
        """
        Generate a summary of sampling results.
        
        Args:
            results (dict): Results from multi_strategy_sampling
            
        Returns:
            dict: Summary statistics
        """
        summary = {}
        
        for strategy, strategy_results in results.items():
            summary[strategy] = {}
            
            for k, result in strategy_results.items():
                metadata = result['metadata']
                n_sampled = metadata['n_sampled']
                sampling_rate = metadata['sampling_rate']
                
                # Compute length distribution preservation
                original_lengths = self.lengths
                sampled_indices = result['indices']
                sampled_lengths = original_lengths[sampled_indices]
                
                length_correlation = np.corrcoef(
                    np.histogram(original_lengths, bins=20)[0],
                    np.histogram(sampled_lengths, bins=20)[0]
                )[0, 1]
                
                summary[strategy][k] = {
                    'n_sampled': n_sampled,
                    'sampling_rate': sampling_rate,
                    'length_distribution_correlation': length_correlation,
                    'mean_length_ratio': np.mean(sampled_lengths) / np.mean(original_lengths)
                }
        
        return summary
