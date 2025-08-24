#!/usr/bin/env python3
"""
Statistics Storage Module
=========================

Comprehensive statistics collection and storage system for tractography
and TDA analysis results. Provides structured pandas DataFrames for
scientific analysis and future reference.
"""

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from .tda_metrics import PersistenceDiagramMetrics, analyze_connectome_differences


class TractographyStatistics:
    """
    Manages comprehensive statistics collection and storage for tractography
    and TDA analyses. Provides structured data storage for comparative studies.
    """
    
    def __init__(self, storage_dir="statistics"):
        """
        Initialize statistics manager.
        
        Args:
            storage_dir (str): Directory to store statistics files
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize empty statistics
        self.tractography_stats = []
        self.tda_stats = []
        self.sampling_stats = []
        self.detailed_tda_data = {}  # Store complete persistence diagrams
        self.metrics_calculator = PersistenceDiagramMetrics()
        
    def add_tractography_analysis(self, dataset_name, file_path, loader_stats, 
                                point_cloud_info, analysis_params):
        """
        Add tractography analysis statistics.
        
        Args:
            dataset_name (str): Name identifier for this dataset
            file_path (str): Path to original TRK file
            loader_stats (dict): Statistics from TractogramLoader
            point_cloud_info (dict): Point cloud creation info
            analysis_params (dict): Analysis parameters used
        """
        stats_entry = {
            'dataset_name': dataset_name,
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'analysis_timestamp': datetime.now().isoformat(),
            
            # Tractography statistics
            'n_streamlines': loader_stats['n_streamlines'],
            'n_points_total': loader_stats['n_points_total'],
            'length_mean': loader_stats['length_mean'],
            'length_std': loader_stats['length_std'],
            'length_min': loader_stats['length_min'],
            'length_max': loader_stats['length_max'],
            
            # Spatial extent
            'x_min': loader_stats['spatial_extent'][0][0],
            'x_max': loader_stats['spatial_extent'][0][1],
            'y_min': loader_stats['spatial_extent'][1][0],
            'y_max': loader_stats['spatial_extent'][1][1],
            'z_min': loader_stats['spatial_extent'][2][0],
            'z_max': loader_stats['spatial_extent'][2][1],
            'spatial_volume': ((loader_stats['spatial_extent'][0][1] - loader_stats['spatial_extent'][0][0]) *
                              (loader_stats['spatial_extent'][1][1] - loader_stats['spatial_extent'][1][0]) *
                              (loader_stats['spatial_extent'][2][1] - loader_stats['spatial_extent'][2][0])),
            
            # Point cloud information
            'initial_points': point_cloud_info['initial_points'],
            'voxel_downsampled_points': point_cloud_info['voxel_downsampled_points'],
            'final_point_cloud_size': point_cloud_info['final_point_cloud_size'],
            'downsample_ratio': point_cloud_info['final_point_cloud_size'] / point_cloud_info['initial_points'],
            
            # Analysis parameters
            'max_points_param': analysis_params['max_points'],
            'voxel_size_param': analysis_params['voxel_size'],
            'max_dimension_param': analysis_params['max_dimension']
        }
        
        self.tractography_stats.append(stats_entry)
        
    def add_tda_analysis(self, dataset_name, tda_analysis, diagrams=None, diagrams_info=None):
        """
        Add TDA analysis statistics with comprehensive metrics.
        
        Args:
            dataset_name (str): Name identifier for this dataset
            tda_analysis (dict): TDA analysis results
            diagrams (list, optional): Raw persistence diagrams
            diagrams_info (dict, optional): Additional diagram information
        """
        stats_entry = {
            'dataset_name': dataset_name,
            'analysis_timestamp': datetime.now().isoformat(),
        }
        
        # Store raw diagrams for distance calculations
        if diagrams is not None:
            self.detailed_tda_data[dataset_name] = {
                'diagrams': diagrams,
                'analysis': tda_analysis
            }
            
            # Compute comprehensive metrics using the new TDA metrics system
            comprehensive_metrics = self.metrics_calculator.compute_basic_metrics(diagrams, dataset_name)
            
            # Add comprehensive metrics to stats
            for key, value in comprehensive_metrics.items():
                if key != 'dataset_name':  # Avoid duplicate
                    stats_entry[key] = value
        
        # Add statistics for each homology dimension (legacy compatibility)
        for dim_name, metrics in tda_analysis.items():
            prefix = dim_name.lower()
            stats_entry.update({
                f'{prefix}_n_features': metrics['n_features'],
                f'{prefix}_n_finite': metrics['n_finite'],
                f'{prefix}_n_infinite': metrics['n_infinite'],
                f'{prefix}_total_persistence': metrics['total_persistence'],
                f'{prefix}_max_persistence': metrics['max_persistence'],
                f'{prefix}_mean_persistence': metrics['mean_persistence']
            })
        
        # Add derived metrics
        if 'H0' in tda_analysis and 'H1' in tda_analysis:
            stats_entry.update({
                'h1_h0_ratio': (tda_analysis['H1']['n_features'] / 
                               max(1, tda_analysis['H0']['n_finite'])),
                'total_persistence_all': (tda_analysis['H0']['total_persistence'] + 
                                        tda_analysis['H1']['total_persistence']),
                'max_persistence_all': max(tda_analysis['H0']['max_persistence'],
                                         tda_analysis['H1']['max_persistence'])
            })
        
        self.tda_stats.append(stats_entry)
    
    def add_sampling_experiment(self, experiment_name, sampling_results, tda_results):
        """
        Add sampling experiment statistics.
        
        Args:
            experiment_name (str): Name of the sampling experiment
            sampling_results (dict): Results from sampling strategies
            tda_results (dict): TDA results for each sample
        """
        for strategy, strategy_results in sampling_results.items():
            for k, sample_result in strategy_results.items():
                metadata = sample_result['metadata']
                tda_analysis = tda_results[strategy][k]['analysis']
                
                stats_entry = {
                    'experiment_name': experiment_name,
                    'sampling_strategy': strategy,
                    'sampling_k': k,
                    'analysis_timestamp': datetime.now().isoformat(),
                    
                    # Sampling metadata
                    'n_original_streamlines': metadata['n_original'],
                    'n_sampled_streamlines': metadata['n_sampled'],
                    'sampling_rate': metadata['sampling_rate'],
                    
                    # TDA results
                    'h0_n_features': tda_analysis['H0']['n_features'],
                    'h0_total_persistence': tda_analysis['H0']['total_persistence'],
                    'h0_max_persistence': tda_analysis['H0']['max_persistence'],
                    'h1_n_features': tda_analysis['H1']['n_features'],
                    'h1_total_persistence': tda_analysis['H1']['total_persistence'],
                    'h1_max_persistence': tda_analysis['H1']['max_persistence'],
                    
                    # Strategy-specific metadata
                    **{f'meta_{k}': v for k, v in metadata.items() 
                       if k not in ['n_original', 'n_sampled', 'sampling_rate']}
                }
                
                self.sampling_stats.append(stats_entry)
    
    def get_tractography_dataframe(self):
        """Get tractography statistics as pandas DataFrame."""
        if not self.tractography_stats:
            return pd.DataFrame()
        return pd.DataFrame(self.tractography_stats)
    
    def get_tda_dataframe(self):
        """Get TDA statistics as pandas DataFrame."""
        if not self.tda_stats:
            return pd.DataFrame()
        return pd.DataFrame(self.tda_stats)
    
    def get_sampling_dataframe(self):
        """Get sampling experiment statistics as pandas DataFrame."""
        if not self.sampling_stats:
            return pd.DataFrame()
        return pd.DataFrame(self.sampling_stats)
    
    def save_all_statistics(self, prefix=""):
        """
        Save all statistics to CSV and JSON files.
        
        Args:
            prefix (str): Optional prefix for output files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if prefix:
            base_name = f"{prefix}_{timestamp}"
        else:
            base_name = f"tractography_statistics_{timestamp}"
        
        # Save DataFrames as CSV
        if self.tractography_stats:
            df_tract = self.get_tractography_dataframe()
            csv_path = os.path.join(self.storage_dir, f"{base_name}_tractography.csv")
            df_tract.to_csv(csv_path, index=False)
            print(f"📊 Saved tractography statistics: {csv_path}")
        
        if self.tda_stats:
            df_tda = self.get_tda_dataframe()
            csv_path = os.path.join(self.storage_dir, f"{base_name}_tda.csv")
            df_tda.to_csv(csv_path, index=False)
            print(f"📊 Saved TDA statistics: {csv_path}")
        
        if self.sampling_stats:
            df_sampling = self.get_sampling_dataframe()
            csv_path = os.path.join(self.storage_dir, f"{base_name}_sampling.csv")
            df_sampling.to_csv(csv_path, index=False)
            print(f"📊 Saved sampling statistics: {csv_path}")
        
        # Save raw data as JSON
        all_stats = {
            'tractography': self.tractography_stats,
            'tda': self.tda_stats,
            'sampling': self.sampling_stats,
            'metadata': {
                'creation_timestamp': timestamp,
                'n_tractography_analyses': len(self.tractography_stats),
                'n_tda_analyses': len(self.tda_stats),
                'n_sampling_experiments': len(self.sampling_stats)
            }
        }
        
        json_path = os.path.join(self.storage_dir, f"{base_name}_complete.json")
        with open(json_path, 'w') as f:
            json.dump(all_stats, f, indent=2, default=str)
        print(f"📊 Saved complete statistics: {json_path}")
        
        return {
            'tractography_df': self.get_tractography_dataframe(),
            'tda_df': self.get_tda_dataframe(),
            'sampling_df': self.get_sampling_dataframe(),
            'files_saved': [csv_path, json_path]
        }
    
    def load_statistics(self, json_file_path):
        """
        Load previously saved statistics.
        
        Args:
            json_file_path (str): Path to JSON statistics file
        """
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        self.tractography_stats = data.get('tractography', [])
        self.tda_stats = data.get('tda', [])
        self.sampling_stats = data.get('sampling', [])
        
        print(f"📊 Loaded statistics from: {json_file_path}")
        print(f"   Tractography analyses: {len(self.tractography_stats)}")
        print(f"   TDA analyses: {len(self.tda_stats)}")
        print(f"   Sampling experiments: {len(self.sampling_stats)}")
    
    def get_comparative_summary(self):
        """Generate a comparative summary of all analyses."""
        if not self.tda_stats:
            return None
            
        df = self.get_tda_dataframe()
        
        summary = {}
        for dataset in df['dataset_name'].unique():
            dataset_data = df[df['dataset_name'] == dataset]
            if len(dataset_data) > 0:
                row = dataset_data.iloc[0]
                summary[dataset] = {
                    'h0_features': row['h0_n_features'],
                    'h1_features': row['h1_n_features'],
                    'h0_max_persistence': row['h0_max_persistence'],
                    'h1_max_persistence': row['h1_max_persistence'],
                    'h1_total_persistence': row['h1_total_persistence']
                }
        
        return pd.DataFrame(summary).T
    
    def compute_connectome_distances(self):
        """
        Compute pairwise distances between all stored connectomes.
        
        Returns:
            pd.DataFrame: Pairwise distance matrix with multiple metrics
        """
        if len(self.detailed_tda_data) < 2:
            print("Warning: Need at least 2 datasets to compute distances")
            return None
        
        print(f"\n🔍 Computing pairwise distances between {len(self.detailed_tda_data)} connectomes...")
        
        # Use the comprehensive distance analysis
        distances_df = self.metrics_calculator.compute_all_distances(
            {name: data['diagrams'] for name, data in self.detailed_tda_data.items()}
        )
        
        return distances_df
    
    def save_comprehensive_analysis(self, prefix="comprehensive"):
        """
        Save complete analysis including pairwise distances and recommendations.
        
        Args:
            prefix (str): Prefix for output files
            
        Returns:
            dict: Summary of saved files and analysis
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{prefix}_{timestamp}"
        
        # Save standard statistics
        standard_results = self.save_all_statistics(prefix=prefix)
        
        # Compute and save distances if we have multiple datasets
        if len(self.detailed_tda_data) >= 2:
            # Perform comprehensive connectome comparison
            analysis_results = analyze_connectome_differences(
                self.detailed_tda_data, 
                save_path=os.path.join(self.storage_dir, base_name)
            )
            
            print(f"\n📊 Comprehensive connectome analysis complete!")
            print(f"   Distance metrics computed for {len(self.detailed_tda_data)} datasets")
            
            return {
                'standard_statistics': standard_results,
                'comprehensive_analysis': analysis_results,
                'n_datasets': len(self.detailed_tda_data),
                'datasets': list(self.detailed_tda_data.keys())
            }
        else:
            print(f"\n📊 Standard statistics saved (no distance analysis - need ≥2 datasets)")
            return {
                'standard_statistics': standard_results,
                'n_datasets': len(self.detailed_tda_data)
            }

    def clear_statistics(self):
        """Clear all stored statistics."""
        self.tractography_stats = []
        self.tda_stats = []
        self.sampling_stats = []
        self.detailed_tda_data = {}


def create_analysis_record(dataset_name, file_path, loader, point_cloud, 
                          analysis, analysis_params):
    """
    Convenience function to create a complete analysis record.
    
    Args:
        dataset_name (str): Dataset identifier
        file_path (str): Path to TRK file
        loader: TractogramLoader instance
        point_cloud (np.ndarray): Point cloud used for TDA
        analysis (dict): TDA analysis results
        analysis_params (dict): Analysis parameters
        
    Returns:
        dict: Complete analysis record
    """
    stats = loader.get_statistics()
    
    return {
        'dataset_name': dataset_name,
        'file_info': {
            'path': file_path,
            'name': os.path.basename(file_path)
        },
        'tractography': stats,
        'point_cloud': {
            'initial_points': stats['n_points_total'],
            'final_point_cloud_size': len(point_cloud),
            'voxel_downsampled_points': len(point_cloud)  # Simplified for this function
        },
        'tda': analysis,
        'parameters': analysis_params,
        'timestamp': datetime.now().isoformat()
    }
