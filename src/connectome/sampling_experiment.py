#!/usr/bin/env python3
"""
Sampling Experiment Module
==========================

Coordinates sampling experiments and TDA analysis to study robustness
and sensitivity of topological features to data sampling.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .sampling import StreamlineSampler
from .topology import analyze_tractography_topology
from .visualization import TractographyVisualizer


class SamplingExperiment:
    """
    Manages comprehensive sampling experiments for TDA robustness studies.
    
    Coordinates sampling strategies, TDA analysis, and results comparison
    to understand how topological features depend on data sampling.
    """
    
    def __init__(self, streamlines, experiment_name="sampling_study", base_output_dir="experiments"):
        """
        Initialize sampling experiment.
        
        Args:
            streamlines (list): Original streamlines
            experiment_name (str): Name for this experiment
            base_output_dir (str): Base directory for outputs
        """
        self.streamlines = streamlines
        self.experiment_name = experiment_name
        self.base_output_dir = base_output_dir
        
        # Create experiment directory structure
        self.experiment_dir = os.path.join(base_output_dir, experiment_name)
        self.results_dir = os.path.join(self.experiment_dir, "results")
        self.figures_dir = os.path.join(self.experiment_dir, "figures")
        self.data_dir = os.path.join(self.experiment_dir, "data")
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize sampler
        self.sampler = StreamlineSampler(streamlines)
        
        # Storage for results
        self.sampling_results = {}
        self.tda_results = {}
        
    def run_sampling_experiment(self, k_values=[5, 10, 20], strategies=None, 
                               tda_params=None):
        """
        Run comprehensive sampling experiment.
        
        Args:
            k_values (list): Sampling factors to test
            strategies (list): Sampling strategies to use
            tda_params (dict): Parameters for TDA analysis
        """
        if tda_params is None:
            tda_params = {
                'max_points': 1500,
                'voxel_size': 2.0,
                'max_dimension': 1
            }
        
        print(f"🔬 Running sampling experiment: {self.experiment_name}")
        print(f"📁 Output directory: {self.experiment_dir}")
        
        # 1. Apply sampling strategies
        print("\n🎲 Applying sampling strategies...")
        self.sampling_results = self.sampler.multi_strategy_sampling(
            k_values=k_values, strategies=strategies
        )
        
        # 2. Run TDA on all samples
        print("\n🔍 Running TDA analysis on samples...")
        self.tda_results = {}
        
        for strategy, strategy_results in self.sampling_results.items():
            print(f"\nStrategy: {strategy}")
            self.tda_results[strategy] = {}
            
            for k, sample_result in strategy_results.items():
                print(f"  k={k}...")
                
                streamlines = sample_result['streamlines']
                metadata = sample_result['metadata']
                
                # Run TDA
                point_cloud, diagrams, analysis = analyze_tractography_topology(
                    streamlines,
                    max_points=tda_params['max_points'],
                    voxel_size=tda_params['voxel_size'],
                    max_dimension=tda_params['max_dimension']
                )
                
                self.tda_results[strategy][k] = {
                    'point_cloud': point_cloud,
                    'diagrams': diagrams,
                    'analysis': analysis,
                    'metadata': metadata
                }
        
        # 3. Save results
        self._save_results()
        
        # 4. Generate visualizations
        self._create_visualizations()
        
        # 5. Generate comparison report
        self._generate_comparison_report()
        
        print(f"\n✅ Experiment complete! Results saved to: {self.experiment_dir}")
    
    def _save_results(self):
        """Save experiment results to files."""
        print("\n💾 Saving results...")
        
        # Save TDA analysis results as JSON (convert numpy types)
        tda_summary = {}
        for strategy, strategy_results in self.tda_results.items():
            tda_summary[strategy] = {}
            for k, result in strategy_results.items():
                # Convert numpy types to Python native types
                analysis = result['analysis']
                converted_analysis = {}
                for dim_name, metrics in analysis.items():
                    converted_analysis[dim_name] = {
                        'n_features': int(metrics['n_features']),
                        'n_finite': int(metrics['n_finite']),
                        'n_infinite': int(metrics['n_infinite']),
                        'total_persistence': float(metrics['total_persistence']),
                        'max_persistence': float(metrics['max_persistence']),
                        'mean_persistence': float(metrics['mean_persistence'])
                    }
                tda_summary[strategy][str(k)] = converted_analysis
        
        with open(os.path.join(self.data_dir, "tda_results.json"), 'w') as f:
            json.dump(tda_summary, f, indent=2)
        
        # Save sampling metadata (convert numpy types)
        sampling_summary = {}
        for strategy, strategy_results in self.sampling_results.items():
            sampling_summary[strategy] = {}
            for k, result in strategy_results.items():
                metadata = result['metadata']
                converted_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, np.ndarray):
                        converted_metadata[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        converted_metadata[key] = value.item()
                    else:
                        converted_metadata[key] = value
                sampling_summary[strategy][str(k)] = converted_metadata
        
        with open(os.path.join(self.data_dir, "sampling_metadata.json"), 'w') as f:
            json.dump(sampling_summary, f, indent=2)
    
    def _create_visualizations(self):
        """Create comprehensive visualizations of sampling experiment."""
        print("\n📊 Creating visualizations...")
        
        visualizer = TractographyVisualizer()
        
        # 1. TDA comparison plots
        self._plot_tda_comparison()
        
        # 2. Sampling strategy comparison
        self._plot_sampling_comparison()
        
        # 3. Robustness analysis
        self._plot_robustness_analysis()
        
        # 4. Individual persistence diagrams for key results
        self._plot_key_persistence_diagrams(visualizer)
    
    def _plot_tda_comparison(self):
        """Plot TDA feature counts across sampling strategies."""
        strategies = list(self.tda_results.keys())
        k_values = list(self.tda_results[strategies[0]].keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'TDA Features vs Sampling Strategy - {self.experiment_name}', fontsize=14)
        
        # Prepare data
        h0_features = {strategy: [] for strategy in strategies}
        h1_features = {strategy: [] for strategy in strategies}
        h0_persistence = {strategy: [] for strategy in strategies}
        h1_persistence = {strategy: [] for strategy in strategies}
        
        for strategy in strategies:
            for k in k_values:
                analysis = self.tda_results[strategy][k]['analysis']
                h0_features[strategy].append(analysis['H0']['n_features'])
                h1_features[strategy].append(analysis['H1']['n_features'])
                h0_persistence[strategy].append(analysis['H0']['total_persistence'])
                h1_persistence[strategy].append(analysis['H1']['total_persistence'])
        
        # Plot H0 features
        ax = axes[0, 0]
        for strategy in strategies:
            ax.plot(k_values, h0_features[strategy], 'o-', label=strategy, linewidth=2)
        ax.set_xlabel('Sampling factor k')
        ax.set_ylabel('H0 features')
        ax.set_title('Connected Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot H1 features
        ax = axes[0, 1]
        for strategy in strategies:
            ax.plot(k_values, h1_features[strategy], 'o-', label=strategy, linewidth=2)
        ax.set_xlabel('Sampling factor k')
        ax.set_ylabel('H1 features')
        ax.set_title('Loops/Holes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot H0 persistence
        ax = axes[1, 0]
        for strategy in strategies:
            ax.plot(k_values, h0_persistence[strategy], 'o-', label=strategy, linewidth=2)
        ax.set_xlabel('Sampling factor k')
        ax.set_ylabel('Total H0 persistence')
        ax.set_title('H0 Total Persistence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot H1 persistence
        ax = axes[1, 1]
        for strategy in strategies:
            ax.plot(k_values, h1_persistence[strategy], 'o-', label=strategy, linewidth=2)
        ax.set_xlabel('Sampling factor k')
        ax.set_ylabel('Total H1 persistence')
        ax.set_title('H1 Total Persistence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'tda_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_sampling_comparison(self):
        """Plot sampling strategy characteristics."""
        strategies = list(self.sampling_results.keys())
        k_values = list(self.sampling_results[strategies[0]].keys())
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Sampling Strategy Comparison - {self.experiment_name}', fontsize=14)
        
        # Sample sizes
        ax = axes[0]
        for strategy in strategies:
            sample_sizes = []
            for k in k_values:
                metadata = self.sampling_results[strategy][k]['metadata']
                sample_sizes.append(metadata['n_sampled'])
            ax.plot(k_values, sample_sizes, 'o-', label=strategy, linewidth=2)
        
        ax.set_xlabel('Sampling factor k')
        ax.set_ylabel('Number of sampled streamlines')
        ax.set_title('Sample Sizes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Sampling rates
        ax = axes[1]
        for strategy in strategies:
            sampling_rates = []
            for k in k_values:
                metadata = self.sampling_results[strategy][k]['metadata']
                sampling_rates.append(metadata['sampling_rate'] * 100)
            ax.plot(k_values, sampling_rates, 'o-', label=strategy, linewidth=2)
        
        ax.set_xlabel('Sampling factor k')
        ax.set_ylabel('Sampling rate (%)')
        ax.set_title('Sampling Rates')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'sampling_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_robustness_analysis(self):
        """Plot robustness metrics showing TDA stability across sampling."""
        strategies = list(self.tda_results.keys())
        k_values = list(self.tda_results[strategies[0]].keys())
        
        # Calculate coefficient of variation for each strategy
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'TDA Robustness Analysis - {self.experiment_name}', fontsize=14)
        
        # H1 feature stability
        ax = axes[0]
        for strategy in strategies:
            h1_counts = []
            for k in k_values:
                analysis = self.tda_results[strategy][k]['analysis']
                h1_counts.append(analysis['H1']['n_features'])
            
            # Calculate coefficient of variation
            if len(h1_counts) > 1:
                cv = np.std(h1_counts) / np.mean(h1_counts) if np.mean(h1_counts) > 0 else 0
                ax.bar(strategy, cv, alpha=0.7)
        
        ax.set_ylabel('Coefficient of Variation')
        ax.set_title('H1 Feature Count Stability\n(Lower = More Robust)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # H1 persistence stability
        ax = axes[1]
        for strategy in strategies:
            h1_persistence = []
            for k in k_values:
                analysis = self.tda_results[strategy][k]['analysis']
                h1_persistence.append(analysis['H1']['total_persistence'])
            
            # Calculate coefficient of variation
            if len(h1_persistence) > 1:
                cv = np.std(h1_persistence) / np.mean(h1_persistence) if np.mean(h1_persistence) > 0 else 0
                ax.bar(strategy, cv, alpha=0.7)
        
        ax.set_ylabel('Coefficient of Variation')
        ax.set_title('H1 Persistence Stability\n(Lower = More Robust)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'robustness_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_key_persistence_diagrams(self, visualizer):
        """Plot persistence diagrams for key sampling results."""
        # Plot diagrams for k=10 for each strategy
        strategies = list(self.tda_results.keys())
        k = 10  # Middle sampling factor
        
        if k not in self.tda_results[strategies[0]]:
            k = list(self.tda_results[strategies[0]].keys())[len(list(self.tda_results[strategies[0]].keys()))//2]
        
        for strategy in strategies:
            if k in self.tda_results[strategy]:
                diagrams = self.tda_results[strategy][k]['diagrams']
                save_path = os.path.join(self.figures_dir, f'persistence_diagrams_{strategy}_k{k}.png')
                
                fig = visualizer.plot_persistence_diagrams(diagrams, save_path=save_path)
                plt.close(fig)
    
    def _generate_comparison_report(self):
        """Generate a comprehensive comparison report."""
        print("\n📋 Generating comparison report...")
        
        # Create summary DataFrame
        summary_data = []
        
        for strategy, strategy_results in self.tda_results.items():
            for k, result in strategy_results.items():
                analysis = result['analysis']
                metadata = result['metadata']
                
                summary_data.append({
                    'strategy': strategy,
                    'k': k,
                    'n_sampled': metadata['n_sampled'],
                    'sampling_rate': metadata['sampling_rate'],
                    'h0_features': analysis['H0']['n_features'],
                    'h1_features': analysis['H1']['n_features'],
                    'h0_total_persistence': analysis['H0']['total_persistence'],
                    'h1_total_persistence': analysis['H1']['total_persistence'],
                    'h0_max_persistence': analysis['H0']['max_persistence'],
                    'h1_max_persistence': analysis['H1']['max_persistence']
                })
        
        df = pd.DataFrame(summary_data)
        
        # Save summary table
        df.to_csv(os.path.join(self.results_dir, 'summary_table.csv'), index=False)
        
        # Generate text report
        report_path = os.path.join(self.results_dir, 'experiment_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Sampling Experiment Report: {self.experiment_name}\n")
            f.write("=" * 60 + "\n\n")
            
            # Original data summary
            f.write(f"Original Data:\n")
            f.write(f"  Total streamlines: {len(self.streamlines)}\n")
            f.write(f"  Sampling strategies tested: {len(df['strategy'].unique())}\n")
            f.write(f"  K values tested: {sorted(df['k'].unique())}\n\n")
            
            # Strategy comparison
            f.write("Strategy Performance Summary:\n")
            f.write("-" * 40 + "\n")
            
            for strategy in df['strategy'].unique():
                strategy_data = df[df['strategy'] == strategy]
                h1_mean = strategy_data['h1_features'].mean()
                h1_std = strategy_data['h1_features'].std()
                
                f.write(f"\n{strategy.upper()}:\n")
                f.write(f"  H1 features: {h1_mean:.1f} ± {h1_std:.1f}\n")
                f.write(f"  H1 stability (CV): {h1_std/h1_mean:.3f}\n")
            
            # Recommendations
            f.write(f"\n\nRecommendations:\n")
            f.write("-" * 20 + "\n")
            
            # Find most robust strategy
            strategy_stability = {}
            for strategy in df['strategy'].unique():
                strategy_data = df[df['strategy'] == strategy]
                h1_cv = strategy_data['h1_features'].std() / strategy_data['h1_features'].mean()
                strategy_stability[strategy] = h1_cv
            
            most_robust = min(strategy_stability.keys(), key=lambda x: strategy_stability[x])
            f.write(f"Most robust strategy: {most_robust} (CV = {strategy_stability[most_robust]:.3f})\n")
            
            # Find optimal k value
            k_stability = {}
            for k in df['k'].unique():
                k_data = df[df['k'] == k]
                h1_cv = k_data['h1_features'].std() / k_data['h1_features'].mean()
                k_stability[k] = h1_cv
            
            optimal_k = min(k_stability.keys(), key=lambda x: k_stability[x])
            f.write(f"Most stable k value: {optimal_k} (CV = {k_stability[optimal_k]:.3f})\n")
        
        print(f"📋 Report saved: {report_path}")
    
    def get_results_summary(self):
        """Return a summary of experiment results."""
        summary = {
            'experiment_name': self.experiment_name,
            'experiment_dir': self.experiment_dir,
            'strategies_tested': list(self.tda_results.keys()),
            'k_values_tested': list(self.tda_results[list(self.tda_results.keys())[0]].keys()),
            'original_streamlines': len(self.streamlines)
        }
        
        return summary
