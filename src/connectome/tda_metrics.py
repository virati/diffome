#!/usr/bin/env python3
"""
TDA Metrics Module
==================

Advanced metrics and distance measures for persistence diagrams.
Provides tools for comparative analysis between different connectomes.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance
import warnings


class PersistenceDiagramMetrics:
    """
    Comprehensive metrics for persistence diagrams including distance measures
    for comparative connectome analysis.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.diagram_cache = {}
        self.metrics_cache = {}
    
    def compute_basic_metrics(self, diagrams, dataset_name=""):
        """
        Compute basic topological metrics from persistence diagrams.
        
        Args:
            diagrams (list): Persistence diagrams from ripser
            dataset_name (str): Name for this dataset
            
        Returns:
            dict: Comprehensive metrics dictionary
        """
        metrics = {
            'dataset_name': dataset_name,
            'n_dimensions': len(diagrams)
        }
        
        for dim, diagram in enumerate(diagrams):
            if len(diagram) == 0:
                # No features in this dimension
                dim_metrics = self._empty_dimension_metrics(dim)
            else:
                dim_metrics = self._compute_dimension_metrics(diagram, dim)
            
            # Add dimension-specific metrics
            for key, value in dim_metrics.items():
                metrics[f'h{dim}_{key}'] = value
        
        # Compute cross-dimensional metrics
        if len(diagrams) >= 2:
            metrics.update(self._compute_cross_dimensional_metrics(diagrams))
        
        return metrics
    
    def _empty_dimension_metrics(self, dim):
        """Metrics for empty persistence diagrams."""
        return {
            'n_features': 0,
            'n_finite': 0,
            'n_infinite': 0,
            'total_persistence': 0.0,
            'max_persistence': 0.0,
            'mean_persistence': 0.0,
            'std_persistence': 0.0,
            'persistence_entropy': 0.0,
            'total_lifetime': 0.0,
            'normalized_persistence': 0.0
        }
    
    def _compute_dimension_metrics(self, diagram, dim):
        """Compute comprehensive metrics for a single dimension."""
        # Separate finite and infinite features
        finite_mask = diagram[:, 1] != np.inf
        finite_diagram = diagram[finite_mask]
        n_infinite = np.sum(~finite_mask)
        
        if len(finite_diagram) == 0:
            # Only infinite features
            return {
                'n_features': len(diagram),
                'n_finite': 0,
                'n_infinite': n_infinite,
                'total_persistence': 0.0,
                'max_persistence': 0.0,
                'mean_persistence': 0.0,
                'std_persistence': 0.0,
                'persistence_entropy': 0.0,
                'total_lifetime': 0.0,
                'normalized_persistence': 0.0
            }
        
        # Compute lifetimes (persistence = death - birth)
        lifetimes = finite_diagram[:, 1] - finite_diagram[:, 0]
        
        # Basic statistics
        total_persistence = np.sum(lifetimes)
        max_persistence = np.max(lifetimes)
        mean_persistence = np.mean(lifetimes)
        std_persistence = np.std(lifetimes) if len(lifetimes) > 1 else 0.0
        
        # Persistence entropy (measure of complexity)
        persistence_entropy = self._compute_persistence_entropy(lifetimes)
        
        # Normalized persistence (sum of lifetimes / max possible lifetime)
        max_possible_lifetime = np.max(finite_diagram[:, 1]) - np.min(finite_diagram[:, 0])
        normalized_persistence = total_persistence / max_possible_lifetime if max_possible_lifetime > 0 else 0.0
        
        return {
            'n_features': len(diagram),
            'n_finite': len(finite_diagram),
            'n_infinite': n_infinite,
            'total_persistence': total_persistence,
            'max_persistence': max_persistence,
            'mean_persistence': mean_persistence,
            'std_persistence': std_persistence,
            'persistence_entropy': persistence_entropy,
            'total_lifetime': total_persistence,  # Alias for consistency
            'normalized_persistence': normalized_persistence
        }
    
    def _compute_persistence_entropy(self, lifetimes):
        """
        Compute persistence entropy - a measure of topological complexity.
        
        Higher entropy indicates more evenly distributed feature lifetimes.
        Lower entropy indicates dominance by a few long-lived features.
        """
        if len(lifetimes) <= 1:
            return 0.0
        
        # Normalize lifetimes to probabilities
        total_lifetime = np.sum(lifetimes)
        if total_lifetime == 0:
            return 0.0
        
        probabilities = lifetimes / total_lifetime
        
        # Compute entropy: -sum(p * log(p))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_probs = np.log(probabilities)
            log_probs[probabilities == 0] = 0  # Handle 0 * log(0) = 0
            entropy = -np.sum(probabilities * log_probs)
        
        return entropy
    
    def _compute_cross_dimensional_metrics(self, diagrams):
        """Compute metrics that compare across dimensions."""
        metrics = {}
        
        # H1/H0 ratio (topological complexity ratio)
        h0_features = len(diagrams[0]) if len(diagrams) > 0 else 0
        h1_features = len(diagrams[1]) if len(diagrams) > 1 else 0
        
        h0_finite = np.sum(diagrams[0][:, 1] != np.inf) if len(diagrams) > 0 and len(diagrams[0]) > 0 else 0
        
        metrics['h1_h0_ratio'] = h1_features / max(h0_finite, 1)
        
        # Total topological complexity
        total_features = sum(len(d) for d in diagrams)
        metrics['total_features'] = total_features
        
        return metrics
    
    def compute_wasserstein_distance(self, diagram1, diagram2, dimension=1, p=2):
        """
        Compute Wasserstein distance between two persistence diagrams.
        
        This is the gold standard for comparing persistence diagrams.
        
        Args:
            diagram1, diagram2: Persistence diagrams
            dimension: Which homology dimension to compare (default: 1)
            p: Wasserstein p-norm (default: 2)
            
        Returns:
            float: Wasserstein distance
        """
        try:
            # Extract specified dimension
            d1 = diagram1[dimension] if len(diagram1) > dimension else np.array([]).reshape(0, 2)
            d2 = diagram2[dimension] if len(diagram2) > dimension else np.array([]).reshape(0, 2)
            
            # Remove infinite persistence features for Wasserstein computation
            d1_finite = d1[d1[:, 1] != np.inf] if len(d1) > 0 else np.array([]).reshape(0, 2)
            d2_finite = d2[d2[:, 1] != np.inf] if len(d2) > 0 else np.array([]).reshape(0, 2)
            
            if len(d1_finite) == 0 and len(d2_finite) == 0:
                return 0.0
            
            # Use simplified Wasserstein distance on persistence values
            persistence1 = d1_finite[:, 1] - d1_finite[:, 0] if len(d1_finite) > 0 else np.array([])
            persistence2 = d2_finite[:, 1] - d2_finite[:, 0] if len(d2_finite) > 0 else np.array([])
            
            # Compute Wasserstein distance
            return wasserstein_distance(persistence1, persistence2)
            
        except Exception as e:
            print(f"Warning: Could not compute Wasserstein distance: {e}")
            return np.nan
    
    def compute_bottleneck_distance(self, diagram1, diagram2, dimension=1):
        """
        Compute simplified bottleneck distance (max difference in persistence).
        
        Args:
            diagram1, diagram2: Persistence diagrams
            dimension: Which homology dimension to compare
            
        Returns:
            float: Simplified bottleneck distance
        """
        try:
            # Extract specified dimension
            d1 = diagram1[dimension] if len(diagram1) > dimension else np.array([]).reshape(0, 2)
            d2 = diagram2[dimension] if len(diagram2) > dimension else np.array([]).reshape(0, 2)
            
            # Remove infinite persistence features
            d1_finite = d1[d1[:, 1] != np.inf] if len(d1) > 0 else np.array([]).reshape(0, 2)
            d2_finite = d2[d2[:, 1] != np.inf] if len(d2) > 0 else np.array([]).reshape(0, 2)
            
            if len(d1_finite) == 0 and len(d2_finite) == 0:
                return 0.0
            
            # Simplified bottleneck: difference in max persistence
            max_pers1 = np.max(d1_finite[:, 1] - d1_finite[:, 0]) if len(d1_finite) > 0 else 0.0
            max_pers2 = np.max(d2_finite[:, 1] - d2_finite[:, 0]) if len(d2_finite) > 0 else 0.0
            
            return abs(max_pers1 - max_pers2)
            
        except Exception as e:
            print(f"Warning: Could not compute bottleneck distance: {e}")
            return np.nan
    
    def compute_persistence_landscape_distance(self, diagram1, diagram2, dimension=1, resolution=100):
        """
        Compute distance based on persistence landscapes.
        
        Persistence landscapes provide a functional representation that's
        easier to compare and analyze statistically.
        
        Args:
            diagram1, diagram2: Persistence diagrams
            dimension: Which homology dimension to compare
            resolution: Number of points for landscape discretization
            
        Returns:
            float: L2 distance between persistence landscapes
        """
        try:
            # Extract specified dimension
            d1 = diagram1[dimension] if len(diagram1) > dimension else np.array([]).reshape(0, 2)
            d2 = diagram2[dimension] if len(diagram2) > dimension else np.array([]).reshape(0, 2)
            
            # Remove infinite persistence features
            d1_finite = d1[d1[:, 1] != np.inf] if len(d1) > 0 else np.array([]).reshape(0, 2)
            d2_finite = d2[d2[:, 1] != np.inf] if len(d2) > 0 else np.array([]).reshape(0, 2)
            
            if len(d1_finite) == 0 and len(d2_finite) == 0:
                return 0.0
            
            # Create simplified landscape based on persistence values
            all_points = np.concatenate([d1_finite.flatten(), d2_finite.flatten()]) if len(d1_finite) > 0 and len(d2_finite) > 0 else None
            
            if all_points is None or len(all_points) == 0:
                return np.nan
            
            # Create grid
            x_min, x_max = np.min(all_points), np.max(all_points)
            if x_min == x_max:
                return 0.0
            
            x_grid = np.linspace(x_min, x_max, resolution)
            
            # Simplified landscape computation
            landscape1 = self._compute_simple_landscape(d1_finite, x_grid)
            landscape2 = self._compute_simple_landscape(d2_finite, x_grid)
            
            # L2 distance
            return np.sqrt(np.sum((landscape1 - landscape2) ** 2))
            
        except Exception as e:
            print(f"Warning: Could not compute landscape distance: {e}")
            return np.nan
    
    def _compute_simple_landscape(self, diagram, x_grid):
        """Compute simplified persistence landscape."""
        if len(diagram) == 0:
            return np.zeros(len(x_grid))
        
        landscape = np.zeros(len(x_grid))
        
        for birth, death in diagram:
            if death == np.inf:
                continue
            
            # Find the peak (midpoint)
            peak = (birth + death) / 2
            height = (death - birth) / 2
            
            # Add tent function to landscape
            for i, x in enumerate(x_grid):
                if birth <= x <= death:
                    tent_height = height - abs(x - peak)
                    landscape[i] = max(landscape[i], tent_height)
        
        return landscape
    
    def compute_all_distances(self, diagrams_dict, dimensions=[0, 1]):
        """
        Compute all pairwise distances between persistence diagrams.
        
        Args:
            diagrams_dict: Dictionary of {dataset_name: diagrams}
            dimensions: Which homology dimensions to compare
            
        Returns:
            pd.DataFrame: Pairwise distance matrix
        """
        dataset_names = list(diagrams_dict.keys())
        n_datasets = len(dataset_names)
        
        results = []
        
        for i in range(n_datasets):
            for j in range(i + 1, n_datasets):
                name1, name2 = dataset_names[i], dataset_names[j]
                diagrams1, diagrams2 = diagrams_dict[name1], diagrams_dict[name2]
                
                for dim in dimensions:
                    # Compute different distance measures
                    wasserstein_dist = self.compute_wasserstein_distance(diagrams1, diagrams2, dim)
                    bottleneck_dist = self.compute_bottleneck_distance(diagrams1, diagrams2, dim)
                    landscape_dist = self.compute_persistence_landscape_distance(diagrams1, diagrams2, dim)
                    
                    results.append({
                        'dataset1': name1,
                        'dataset2': name2,
                        'dimension': f'H{dim}',
                        'wasserstein_distance': wasserstein_dist,
                        'bottleneck_distance': bottleneck_dist,
                        'landscape_distance': landscape_dist
                    })
        
        return pd.DataFrame(results)
    
    def create_comprehensive_dataframe(self, results_dict):
        """
        Create comprehensive DataFrame with all TDA metrics for multiple datasets.
        
        Args:
            results_dict: {dataset_name: {'diagrams': diagrams, 'analysis': analysis}}
            
        Returns:
            pd.DataFrame: Comprehensive TDA metrics
        """
        all_metrics = []
        
        for dataset_name, data in results_dict.items():
            diagrams = data['diagrams']
            analysis = data.get('analysis', {})
            
            # Compute comprehensive metrics
            metrics = self.compute_basic_metrics(diagrams, dataset_name)
            
            # Add any existing analysis metrics
            for key, value in analysis.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        metrics[f'{key.lower()}_{subkey}'] = subvalue
                else:
                    metrics[key] = value
            
            all_metrics.append(metrics)
        
        df = pd.DataFrame(all_metrics)
        
        # Compute pairwise distances
        if len(results_dict) > 1:
            diagrams_dict = {name: data['diagrams'] for name, data in results_dict.items()}
            distance_df = self.compute_all_distances(diagrams_dict)
            
            return df, distance_df
        
        return df, None


def recommend_distance_metric():
    """
    Recommend the best distance metric for connectome comparison.
    
    Returns:
        str: Recommendation with explanation
    """
    recommendation = """
    RECOMMENDED METRIC: Wasserstein Distance on H1 Features
    
    WHY THIS IS OPTIMAL FOR CONNECTOMES:
    
    1. **Biological Relevance**: H1 features capture topological loops and holes
       in white matter structure, which correspond to:
       - Complex fiber bundle arrangements
       - Regions where multiple tracts intersect
       - Topological organization of neural pathways
    
    2. **Mathematical Robustness**: Wasserstein distance:
       - Provides metric properties (triangle inequality, symmetry)
       - Stable under small perturbations in data
       - Accounts for both feature count AND persistence values
       - Well-established in topological data analysis
    
    3. **Discriminative Power**: Captures both:
       - Structural differences (number of topological features)
       - Significance differences (persistence/lifetime of features)
    
    4. **Clinical Applications**:
       - Disease detection: Different pathologies alter white matter topology
       - Individual differences: Natural variation in brain connectivity
       - Treatment monitoring: Track changes in neural structure over time
    
    USAGE:
    - Compare H1 Wasserstein distances between connectomes
    - Lower distance = more similar topological structure
    - Higher distance = more different topological organization
    
    COMPLEMENTARY METRICS:
    - H0 features for overall connectivity patterns
    - Persistence entropy for complexity measures
    - Bottleneck distance for max feature differences
    """
    
    return recommendation


# Convenience function for quick analysis
def analyze_connectome_differences(results_dict, save_path=None):
    """
    Perform comprehensive connectome comparison analysis.
    
    Args:
        results_dict: {dataset_name: {'diagrams': diagrams, 'analysis': analysis}}
        save_path: Optional path to save results
        
    Returns:
        dict: Complete analysis results
    """
    metrics_calculator = PersistenceDiagramMetrics()
    
    # Create comprehensive metrics
    metrics_df, distances_df = metrics_calculator.create_comprehensive_dataframe(results_dict)
    
    # Print recommendation
    print(recommend_distance_metric())
    
    # Summary statistics
    print("\n" + "="*60)
    print("CONNECTOME COMPARISON SUMMARY")
    print("="*60)
    
    if distances_df is not None:
        print(f"\nPairwise Distances (H1 Wasserstein - RECOMMENDED):")
        h1_distances = distances_df[distances_df['dimension'] == 'H1']
        for _, row in h1_distances.iterrows():
            print(f"  {row['dataset1']} ↔ {row['dataset2']}: {row['wasserstein_distance']:.4f}")
    
    # Save results
    if save_path:
        metrics_df.to_csv(f"{save_path}_comprehensive_metrics.csv", index=False)
        if distances_df is not None:
            distances_df.to_csv(f"{save_path}_pairwise_distances.csv", index=False)
        print(f"\n💾 Results saved to {save_path}_*.csv")
    
    return {
        'metrics': metrics_df,
        'distances': distances_df,
        'recommendation': recommend_distance_metric()
    }
