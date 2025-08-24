#!/usr/bin/env python3
"""
Joint Analysis Module
====================

Performs combined analysis of multiple tractography datasets to understand
joint topological properties and comparative features.
"""

import numpy as np
from .tractography import TractogramLoader
from .topology import analyze_tractography_topology
from .visualization import TractographyVisualizer
from .connectome_statistics import TractographyStatistics, create_analysis_record


class JointTractographyAnalysis:
    """
    Performs joint analysis of multiple tractography datasets.
    
    Combines streamlines from multiple sources and analyzes the joint
    topological properties while maintaining individual dataset tracking.
    """
    
    def __init__(self, dataset_configs, output_prefix="pd_joint"):
        """
        Initialize joint analysis.
        
        Args:
            dataset_configs (list): List of dicts with 'name' and 'file_path'
            output_prefix (str): Prefix for output files
        """
        self.dataset_configs = dataset_configs
        self.output_prefix = output_prefix
        
        # Storage for loaded data
        self.individual_loaders = {}
        self.individual_stats = {}
        self.combined_streamlines = []
        self.streamline_sources = []  # Track which dataset each streamline comes from
        
        # Analysis results
        self.joint_analysis_results = None
        self.statistics_manager = TractographyStatistics()
        
    def load_all_datasets(self):
        """Load all individual datasets."""
        print(f"🔗 Loading datasets for joint analysis:")
        
        total_streamlines = 0
        
        for config in self.dataset_configs:
            name = config['name']
            file_path = config['file_path']
            
            print(f"  📁 Loading {name}: {file_path}")
            
            # Load individual dataset
            loader = TractogramLoader(file_path)
            loader.load()
            
            # Store individual results
            self.individual_loaders[name] = loader
            self.individual_stats[name] = loader.get_statistics()
            
            # Add to combined streamlines
            streamlines = loader.streamlines
            self.combined_streamlines.extend(streamlines)
            
            # Track source dataset for each streamline
            self.streamline_sources.extend([name] * len(streamlines))
            
            total_streamlines += len(streamlines)
            print(f"    ✓ {len(streamlines):,} streamlines loaded")
        
        print(f"\n📊 Joint dataset summary:")
        print(f"  Total streamlines: {total_streamlines:,}")
        print(f"  Datasets combined: {len(self.dataset_configs)}")
        
        return self
    
    def run_joint_analysis(self, max_points=1500, voxel_size=2.0, max_dimension=1):
        """
        Run TDA analysis on the combined dataset.
        
        Args:
            max_points (int): Maximum points for TDA
            voxel_size (float): Voxel size for downsampling
            max_dimension (int): Maximum homology dimension
        """
        if not self.combined_streamlines:
            raise ValueError("Must load datasets first")
        
        print(f"\n🔬 Running joint topological analysis...")
        print(f"  Combined streamlines: {len(self.combined_streamlines):,}")
        
        # Run TDA on combined streamlines
        point_cloud, diagrams, analysis = analyze_tractography_topology(
            self.combined_streamlines,
            max_points=max_points,
            voxel_size=voxel_size,
            max_dimension=max_dimension
        )
        
        self.joint_analysis_results = {
            'point_cloud': point_cloud,
            'diagrams': diagrams,
            'analysis': analysis,
            'parameters': {
                'max_points': max_points,
                'voxel_size': voxel_size,
                'max_dimension': max_dimension
            }
        }
        
        # Add to statistics
        joint_stats = {
            'n_streamlines': len(self.combined_streamlines),
            'n_points_total': sum(len(s) for s in self.combined_streamlines),
            'length_mean': np.mean([len(s) for s in self.combined_streamlines]),
            'length_std': np.std([len(s) for s in self.combined_streamlines]),
            'length_min': min(len(s) for s in self.combined_streamlines),
            'length_max': max(len(s) for s in self.combined_streamlines),
            'spatial_extent': self._compute_joint_spatial_extent()
        }
        
        self.statistics_manager.add_tractography_analysis(
            dataset_name=self.output_prefix,
            file_path="joint_analysis",
            loader_stats=joint_stats,
            point_cloud_info={
                'initial_points': joint_stats['n_points_total'],
                'voxel_downsampled_points': len(point_cloud),
                'final_point_cloud_size': len(point_cloud)
            },
            analysis_params={
                'max_points': max_points,
                'voxel_size': voxel_size,
                'max_dimension': max_dimension
            }
        )
        
        self.statistics_manager.add_tda_analysis(
            dataset_name=self.output_prefix,
            tda_analysis=analysis,
            diagrams=diagrams
        )
        
        return self.joint_analysis_results
    
    def _compute_joint_spatial_extent(self):
        """Compute spatial extent of combined streamlines."""
        if not self.combined_streamlines:
            return [(0, 0), (0, 0), (0, 0)]
        
        all_points = np.concatenate([s for s in self.combined_streamlines])
        
        return [
            (all_points[:, 0].min(), all_points[:, 0].max()),
            (all_points[:, 1].min(), all_points[:, 1].max()),
            (all_points[:, 2].min(), all_points[:, 2].max())
        ]
    
    def create_visualizations(self, output_dir="outputs"):
        """Create comprehensive visualizations for joint analysis."""
        if self.joint_analysis_results is None:
            raise ValueError("Must run joint analysis first")
        
        print(f"\n📊 Creating joint analysis visualizations...")
        
        visualizer = TractographyVisualizer()
        
        # Joint streamline visualization
        visualizer.plot_streamlines(
            self.combined_streamlines,
            n_sample=100,
            save_path=f'{output_dir}/{self.output_prefix}_streamlines.png'
        )
        
        # Joint point cloud visualization
        visualizer.plot_point_cloud(
            self.joint_analysis_results['point_cloud'],
            save_path=f'{output_dir}/{self.output_prefix}_point_cloud.png'
        )
        
        # Joint persistence diagrams
        visualizer.plot_persistence_diagrams(
            self.joint_analysis_results['diagrams'],
            save_path=f'{output_dir}/{self.output_prefix}_persistence_diagrams.png'
        )
        
        # Joint Betti curves
        visualizer.plot_betti_curves(
            self.joint_analysis_results['diagrams'],
            save_path=f'{output_dir}/{self.output_prefix}_betti_curves.png'
        )
        
        print(f"✓ Joint visualizations saved with prefix: {self.output_prefix}")
    
    def create_comparative_analysis(self):
        """Create comparative analysis between individual and joint results."""
        if self.joint_analysis_results is None:
            raise ValueError("Must run joint analysis first")
        
        print(f"\n📈 Creating comparative analysis...")
        
        # Add individual analyses to statistics
        for name, loader in self.individual_loaders.items():
            print(f"Running individual analysis for {name}...")
            # Run individual TDA for comparison
            try:
                point_cloud, diagrams, analysis = analyze_tractography_topology(
                    loader.streamlines,
                    **self.joint_analysis_results['parameters']
                )
                
                # Add to statistics
                stats = loader.get_statistics()
                self.statistics_manager.add_tractography_analysis(
                    dataset_name=name,
                    file_path=loader.file_path,
                    loader_stats=stats,
                    point_cloud_info={
                        'initial_points': stats['n_points_total'],
                        'voxel_downsampled_points': len(point_cloud),
                        'final_point_cloud_size': len(point_cloud)
                    },
                    analysis_params=self.joint_analysis_results['parameters']
                )
                
                self.statistics_manager.add_tda_analysis(
                    dataset_name=name,
                    tda_analysis=analysis,
                    diagrams=diagrams
                )
                
            except Exception as e:
                print(f"Error in individual analysis for {name}: {e}")
                continue
        
        # Generate comparative summary
        summary = self.statistics_manager.get_comparative_summary()
        print(f"\n📋 Comparative TDA Summary:")
        print(summary)
        
        return summary
    
    def save_comprehensive_statistics(self):
        """Save comprehensive statistics including joint and individual analyses."""
        return self.statistics_manager.save_comprehensive_analysis(prefix=self.output_prefix)
    
    def get_joint_summary(self):
        """Get summary of joint analysis results."""
        if self.joint_analysis_results is None:
            return None
        
        analysis = self.joint_analysis_results['analysis']
        
        return {
            'datasets_combined': [config['name'] for config in self.dataset_configs],
            'total_streamlines': len(self.combined_streamlines),
            'total_points': sum(len(s) for s in self.combined_streamlines),
            'h0_features': analysis['H0']['n_features'],
            'h1_features': analysis['H1']['n_features'],
            'h0_max_persistence': analysis['H0']['max_persistence'],
            'h1_max_persistence': analysis['H1']['max_persistence'],
            'h1_total_persistence': analysis['H1']['total_persistence']
        }


def run_joint_analysis(dataset_configs, output_prefix="pd_joint", 
                      analysis_params=None, create_visualizations=True):
    """
    Convenience function to run complete joint analysis.
    
    Args:
        dataset_configs (list): List of dicts with 'name' and 'file_path'
        output_prefix (str): Prefix for output files
        analysis_params (dict): TDA analysis parameters
        create_visualizations (bool): Whether to create visualizations
        
    Returns:
        JointTractographyAnalysis: Configured analysis object with results
    """
    if analysis_params is None:
        analysis_params = {
            'max_points': 1500,
            'voxel_size': 2.0,
            'max_dimension': 1
        }
    
    # Initialize joint analysis
    joint_analysis = JointTractographyAnalysis(dataset_configs, output_prefix)
    
    # Load all datasets
    joint_analysis.load_all_datasets()
    
    # Run joint TDA
    joint_analysis.run_joint_analysis(**analysis_params)
    
    # Create comparative analysis
    joint_analysis.create_comparative_analysis()
    
    # Create visualizations
    if create_visualizations:
        joint_analysis.create_visualizations()
    
    # Save statistics
    stats_files = joint_analysis.save_comprehensive_statistics()
    
    # Print summary
    summary = joint_analysis.get_joint_summary()
    print(f"\n🎯 Joint Analysis Complete!")
    print(f"  Datasets: {', '.join(summary['datasets_combined'])}")
    print(f"  Total streamlines: {summary['total_streamlines']:,}")
    print(f"  H0 features: {summary['h0_features']}")
    print(f"  H1 features: {summary['h1_features']}")
    print(f"  H1 max persistence: {summary['h1_max_persistence']:.3f}")
    
    return joint_analysis
