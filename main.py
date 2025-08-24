#!/usr/bin/env python3
"""
Connectome Analysis - Main Script
=================================

Clean tractography analysis and topological data analysis pipeline.

Usage:
    python main.py

This script performs:
1. Load TRK tractography file
2. Basic streamline analysis and statistics
3. Create visualizations (streamlines and point cloud)
4. Run persistent homology analysis
5. Generate summary reports

Dependencies:
    pip install dipy nibabel numpy matplotlib ripser
"""

import sys
import os
import argparse
from src.connectome.tractography import TractogramLoader, find_tractography_file
from src.connectome.topology import analyze_tractography_topology
from src.connectome.visualization import TractographyVisualizer
from src.connectome.connectome_statistics import TractographyStatistics
from src.connectome.joint_analysis import run_joint_analysis


# Topological Data Analysis Parameters
MAXIMUM_POINT_CLOUD_SIZE = 5000      # Maximum points for TDA (computational vs accuracy tradeoff)
VOXEL_DOWNSAMPLING_SIZE_MM = 2.0     # Voxel size for spatial downsampling (millimeters)
STREAMLINES_VISUALIZATION_COUNT = 100 # Number of streamlines to display in visualizations
MAXIMUM_HOMOLOGY_DIMENSION = 1       # Maximum homology dimension to compute (0=components, 1=loops)


def main():
    """Main analysis pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tractography Topological Analysis')
    parser.add_argument('--file', '-f', type=str, help='Path to TRK file (optional)')
    parser.add_argument('--output-prefix', '-o', type=str, default='', 
                       help='Prefix for output files (optional)')
    parser.add_argument('--sampling-experiment', action='store_true',
                       help='Run sampling robustness experiment')
    parser.add_argument('--sampling-k', type=int, nargs='+', default=[5, 10, 20],
                       help='Sampling factors for experiment (default: 5 10 20)')
    parser.add_argument('--experiment-name', type=str, default='default_sampling_study',
                       help='Name for sampling experiment')
    parser.add_argument('--joint-analysis', action='store_true',
                       help='Run joint analysis on all available PD files')
    parser.add_argument('--save-statistics', action='store_true',
                       help='Save comprehensive statistics to DataFrame files')
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRACTOGRAPHY TOPOLOGICAL ANALYSIS")
    print("=" * 60)
    
    # 1. Find and load tractography file
    if args.file:
        trk_path = args.file
        if not trk_path.endswith('.trk'):
            print("❌ ERROR: File must be a .trk file")
            sys.exit(1)
    else:
        trk_path = find_tractography_file()
        if trk_path is None:
            print("❌ ERROR: No TRK file found!")
            print("Place a .trk file in the current directory or specify with --file")
            sys.exit(1)
    
    print(f"📁 Loading: {trk_path}")
    
    # Set output file prefix if provided
    prefix = args.output_prefix
    if prefix and not prefix.endswith('_'):
        prefix += '_'
    
    try:
        # Initialize statistics manager
        stats_manager = TractographyStatistics() if args.save_statistics else None
        
        if args.joint_analysis:
            # Run joint analysis on all PD files
            print(f"\n🔗 Running joint analysis...")
            
            # Find all Petersen dataset TRK files
            petersen_dataset_files = []
            for filename in ["data/Petersen PD Top Left.trk", "data/Peterson PD Top Right.trk"]:
                if os.path.exists(filename):
                    # Create clear dataset identifier
                    if "Left" in filename:
                        dataset_id = "petersen_left_hemisphere"
                    elif "Right" in filename:
                        dataset_id = "petersen_right_hemisphere"
                    else:
                        dataset_id = filename.replace('.trk', '').replace(' ', '_').lower()
                    
                    petersen_dataset_files.append({
                        'name': dataset_id,
                        'file_path': filename
                    })
            
            if len(petersen_dataset_files) < 2:
                print("❌ ERROR: Need at least 2 Petersen dataset files for joint analysis")
                print("Expected: 'data/Petersen PD Top Left.trk' and 'data/Peterson PD Top Right.trk'")
                sys.exit(1)
            
            # Run joint analysis
            joint_analysis = run_joint_analysis(
                dataset_configs=petersen_dataset_files,
                output_prefix="petersen_bilateral_combined",
                analysis_params={
                    'max_points': MAXIMUM_POINT_CLOUD_SIZE,
                    'voxel_size': VOXEL_DOWNSAMPLING_SIZE_MM,
                    'max_dimension': MAXIMUM_HOMOLOGY_DIMENSION
                }
            )
            
            print(f"\n✅ Joint analysis complete!")
            
        elif args.sampling_experiment:
            # Load tractography for sampling experiment
            loader = TractogramLoader(trk_path)
            loader.load()
            loader.print_summary()
            
            # Run sampling experiment
            from src.connectome.sampling_experiment import SamplingExperiment
            
            print(f"\n🎲 Running sampling robustness experiment...")
            print(f"   Experiment name: {args.experiment_name}")
            print(f"   Sampling factors: {args.sampling_k}")
            
            experiment = SamplingExperiment(
                loader.streamlines, 
                experiment_name=args.experiment_name
            )
            
            experiment.run_sampling_experiment(
                k_values=args.sampling_k,
                tda_params={
                    'max_points': MAXIMUM_POINT_CLOUD_SIZE,
                    'voxel_size': VOXEL_DOWNSAMPLING_SIZE_MM,
                    'max_dimension': MAXIMUM_HOMOLOGY_DIMENSION
                }
            )
            
            summary = experiment.get_results_summary()
            print(f"\n📊 Experiment completed!")
            print(f"   Results directory: {summary['experiment_dir']}")
            print(f"   Strategies tested: {len(summary['strategies_tested'])}")
            print(f"   K values tested: {summary['k_values_tested']}")
            
        else:
            # Standard single analysis
            loader = TractogramLoader(trk_path)
            loader.load()
            loader.print_summary()
            
            # Ensure outputs directory exists
            os.makedirs('outputs', exist_ok=True)
            
            # Determine dataset identifier for statistics and outputs
            if prefix:
                if 'top_left' in prefix.lower():
                    dataset_identifier = 'petersen_left_hemisphere'
                elif 'top_right' in prefix.lower():
                    dataset_identifier = 'petersen_right_hemisphere'
                else:
                    dataset_identifier = prefix.rstrip('_')
            else:
                dataset_identifier = 'default_analysis'
            
            print(f"\n📊 Creating visualizations...")
            tractography_visualizer = TractographyVisualizer()
            
            # Plot streamlines with clear naming
            tractography_visualizer.plot_streamlines(
                loader.streamlines, 
                n_sample=STREAMLINES_VISUALIZATION_COUNT,
                save_path=f'outputs/{dataset_identifier}_streamlines_visualization.png'
            )
            
            print(f"\n🔬 Running topological data analysis...")
            downsampled_point_cloud, persistence_diagrams, topological_feature_analysis = analyze_tractography_topology(
                loader.streamlines,
                max_points=MAXIMUM_POINT_CLOUD_SIZE,
                voxel_size=VOXEL_DOWNSAMPLING_SIZE_MM,
                max_dimension=MAXIMUM_HOMOLOGY_DIMENSION
            )
            
            print(f"\n📈 Creating topological data analysis visualizations...")
            
            # Plot point cloud projection
            tractography_visualizer.plot_point_cloud(
                downsampled_point_cloud, 
                save_path=f'outputs/{dataset_identifier}_point_cloud_projection.png'
            )
            
            # Plot persistence diagrams
            tractography_visualizer.plot_persistence_diagrams(
                persistence_diagrams, 
                save_path=f'outputs/{dataset_identifier}_persistence_diagrams.png'
            )
            
            # Plot Betti curves
            tractography_visualizer.plot_betti_curves(
                persistence_diagrams, 
                save_path=f'outputs/{dataset_identifier}_betti_curves.png'
            )
            
            # Save statistics if requested
            if stats_manager:
                print(f"\n💾 Saving statistics...")
                loader_stats = loader.get_statistics()
                
                stats_manager.add_tractography_analysis(
                    dataset_name=dataset_identifier,
                    file_path=trk_path,
                    loader_stats=loader_stats,
                    point_cloud_info={
                        'initial_points': loader_stats['n_points_total'],
                        'voxel_downsampled_points': len(downsampled_point_cloud),
                        'final_point_cloud_size': len(downsampled_point_cloud)
                    },
                    analysis_params={
                        'max_points': MAXIMUM_POINT_CLOUD_SIZE,
                        'voxel_size': VOXEL_DOWNSAMPLING_SIZE_MM,
                        'max_dimension': MAXIMUM_HOMOLOGY_DIMENSION
                    }
                )
                
                stats_manager.add_tda_analysis(
                    dataset_name=dataset_identifier,
                    tda_analysis=topological_feature_analysis,
                    diagrams=persistence_diagrams
                )
                
                try:
                    stats_files = stats_manager.save_comprehensive_analysis(prefix=dataset_identifier)
                    print(f"✅ Statistics saved successfully")
                except Exception as e:
                    print(f"Warning: Could not save statistics: {e}")
                    import traceback
                    traceback.print_exc()
            
            print("\n" + "=" * 60)
            print("✅ ANALYSIS COMPLETE")
            print("=" * 60)
            
            print(f"\n📋 Analysis Summary:")
            tractography_stats = loader.get_statistics()
            print(f"  • {tractography_stats['n_streamlines']:,} white matter streamlines analyzed")
            print(f"  • {len(downsampled_point_cloud):,} points used for topological analysis")
            
            print(f"\n🔍 Topological Features Detected:")
            for dimension_name, feature_metrics in topological_feature_analysis.items():
                feature_count = feature_metrics['n_features']
                if feature_count > 0:
                    print(f"  • {dimension_name}: {feature_count} features "
                          f"(max persistence: {feature_metrics['max_persistence']:.3f})")
                else:
                    print(f"  • {dimension_name}: No features detected")
            
            print(f"\n📁 Generated Visualization Files:")
            print(f"  • {dataset_identifier}_streamlines_visualization.png - 3D streamline plots")
            print(f"  • {dataset_identifier}_point_cloud_projection.png - Downsampled point cloud")
            print(f"  • {dataset_identifier}_persistence_diagrams.png - Topological feature birth/death")
            print(f"  • {dataset_identifier}_betti_curves.png - Feature evolution across scales")
            
            if stats_manager:
                print(f"  • statistics/{dataset_identifier}_*.csv - Comprehensive statistical data")
            
            print(f"\n💡 Interpretation:")
            print(f"  • H0: Connected components (separate fiber bundles)")
            print(f"  • H1: Loops/holes in white matter structure")
            print(f"  • Higher persistence = more significant topological features")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"\n🔧 Troubleshooting:")
        print(f"  1. Verify TRK file is valid")
        print(f"  2. Install dependencies: pip install -r requirements.txt")
        print(f"  3. Check available memory (reduce MAX_POINTS if needed)")
        sys.exit(1)


if __name__ == "__main__":
    main()
