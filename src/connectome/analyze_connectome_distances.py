#!/usr/bin/env python3
"""
Connectome Distance Analysis
============================

Demonstrates comprehensive TDA metrics and persistence diagram distance measures
for comparing different connectomes. This script provides the recommended 
approach for connectome comparison using topological data analysis.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

from .tractography import TractogramLoader
from .topology import analyze_tractography_topology
from .tda_metrics import PersistenceDiagramMetrics, analyze_connectome_differences, recommend_distance_metric
from .connectome_statistics import TractographyStatistics


def main():
    """Run comprehensive connectome distance analysis."""
    
    print("="*80)
    print("CONNECTOME TOPOLOGICAL DISTANCE ANALYSIS")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Analysis parameters
    MAX_POINTS = 5000
    VOXEL_SIZE = 2.0
    MAX_DIMENSION = 1
    
    # Petersen Dataset Configurations
    petersen_datasets = [
        {
            "name": "petersen_left_hemisphere", 
            "file": "data/Petersen PD Top Left.trk",
            "hemisphere": "left"
        },
        {
            "name": "petersen_right_hemisphere", 
            "file": "data/Peterson PD Top Right.trk",
            "hemisphere": "right"
        }
    ]
    
    print(f"\n📊 Analyzing {len(petersen_datasets)} Petersen connectome datasets...")
    
    # Store results for topological distance analysis
    tda_analysis_results = {}
    
    # Process each hemisphere dataset
    for dataset_config in petersen_datasets:
        dataset_identifier = dataset_config["name"]
        trk_file_path = dataset_config["file"]
        hemisphere_type = dataset_config["hemisphere"]
        
        print(f"\n🧠 Processing {dataset_identifier}...")
        print(f"   File: {trk_file_path}")
        print(f"   Hemisphere: {hemisphere_type}")
        
        if not os.path.exists(trk_file_path):
            print(f"   ❌ File not found: {trk_file_path}")
            continue
        
        # Load tractography dataset
        tractography_loader = TractogramLoader(trk_file_path)
        tractography_loader.load()
        
        streamline_count = tractography_loader.get_statistics()['n_streamlines']
        print(f"   ✓ Loaded {streamline_count:,} white matter streamlines")
        
        # Run topological data analysis
        downsampled_point_cloud, persistence_diagrams, topological_analysis = analyze_tractography_topology(
            tractography_loader.streamlines,
            max_points=MAX_POINTS,
            voxel_size=VOXEL_SIZE,
            max_dimension=MAX_DIMENSION
        )
        
        # Store comprehensive results
        tda_analysis_results[dataset_identifier] = {
            'persistence_diagrams': persistence_diagrams,
            'topological_analysis': topological_analysis,
            'downsampled_point_cloud': downsampled_point_cloud,
            'tractography_statistics': tractography_loader.get_statistics(),
            'hemisphere_type': hemisphere_type
        }
        
        h0_feature_count = topological_analysis['H0']['n_features']
        h1_feature_count = topological_analysis['H1']['n_features']
        print(f"   ✓ TDA complete: H0={h0_feature_count}, H1={h1_feature_count}")
    
    # Comprehensive distance analysis
    print(f"\n🔍 COMPUTING PERSISTENCE DIAGRAM DISTANCES")
    print("="*60)
    
    if len(tda_analysis_results) >= 2:
        # Prepare data for comprehensive distance analysis
        connectome_distance_input = {
            identifier: {
                'diagrams': data['persistence_diagrams'],
                'analysis': data['topological_analysis']
            } 
            for identifier, data in tda_analysis_results.items()
        }
        
        # Run comprehensive connectome comparison analysis
        connectome_comparison_results = analyze_connectome_differences(
            connectome_distance_input, 
            save_path="statistics/petersen_hemisphere_distance_analysis"
        )
        
        # Display results
        print(f"\n📈 DISTANCE ANALYSIS RESULTS:")
        print("="*50)
        
        pairwise_distances_df = connectome_comparison_results['distances']
        if pairwise_distances_df is not None:
            # Display H1 Wasserstein distances (recommended for connectome comparison)
            h1_wasserstein_distances = pairwise_distances_df[pairwise_distances_df['dimension'] == 'H1']
            
            print(f"\n🎯 H1 WASSERSTEIN DISTANCES (RECOMMENDED FOR CONNECTOMES):")
            for _, distance_row in h1_wasserstein_distances.iterrows():
                hemisphere1, hemisphere2 = distance_row['dataset1'], distance_row['dataset2']
                wasserstein_distance_value = distance_row['wasserstein_distance']
                print(f"   {hemisphere1} ↔ {hemisphere2}: {wasserstein_distance_value:.6f}")
            
            print(f"\n📊 COMPLETE DISTANCE METRICS TABLE:")
            print(pairwise_distances_df.round(6))
        
        # Detailed topological metrics comparison
        comprehensive_metrics_df = connectome_comparison_results['metrics']
        print(f"\n📋 DETAILED TDA METRICS COMPARISON:")
        
        # Select key metrics for comparison
        key_metrics = [
            'dataset_name', 'h0_n_features', 'h1_n_features', 
            'h1_max_persistence', 'h1_total_persistence', 'h1_persistence_entropy'
        ]
        
        if all(col in comprehensive_metrics_df.columns for col in key_metrics):
            comparison_df = comprehensive_metrics_df[key_metrics].round(4)
            print(comparison_df.to_string(index=False))
        
        # Interpretation
        print(f"\n💡 BIOLOGICAL INTERPRETATION:")
        print("="*50)
        
        if len(h1_wasserstein_distances) > 0:
            h1_distance_value = h1_wasserstein_distances.iloc[0]['wasserstein_distance']
            
            if h1_distance_value < 0.1:
                biological_interpretation = "Very similar white matter topology"
            elif h1_distance_value < 0.5:
                biological_interpretation = "Moderately similar topology with some differences"
            elif h1_distance_value < 1.0:
                biological_interpretation = "Notable topological differences"
            else:
                biological_interpretation = "Substantially different white matter organization"
            
            print(f"   Hemispheric comparison: {biological_interpretation}")
            print(f"   H1 Wasserstein distance: {h1_distance_value:.6f}")
        
        # Research recommendations
        print(f"\n🔬 RESEARCH APPLICATIONS:")
        print("="*40)
        print("1. 🧭 Disease Detection: Compare patient vs. healthy connectomes")
        print("2. 🧬 Individual Differences: Quantify natural variation in brain topology")
        print("3. 📈 Treatment Monitoring: Track changes in white matter structure")
        print("4. 🎯 Biomarker Development: Use H1 persistence for clinical markers")
        print("5. 🔄 Longitudinal Studies: Monitor topological changes over time")
        
    else:
        print(f"\n⚠️  Need at least 2 datasets for distance analysis")
        print(f"   Found: {len(tda_analysis_results)} datasets")
    
    # Save comprehensive statistics
    print(f"\n💾 SAVING COMPREHENSIVE STATISTICS")
    print("="*50)
    
    comprehensive_statistics_manager = TractographyStatistics()
    
    # Add all analyses to statistics
    for dataset_identifier, analysis_data in tda_analysis_results.items():
        # Skip tractography statistics for now (focus on TDA metrics)
        
        # Add TDA statistics with persistence diagrams
        comprehensive_statistics_manager.add_tda_analysis(
            dataset_name=dataset_identifier,
            tda_analysis=analysis_data['topological_analysis'],
            diagrams=analysis_data['persistence_diagrams']
        )
    
    # Save comprehensive analysis
    try:
        results = comprehensive_statistics_manager.save_comprehensive_analysis(prefix="petersen_hemisphere_comprehensive_analysis")
        print(f"✅ Comprehensive statistics saved successfully")
        print(f"   Files saved in: statistics/")
    except Exception as e:
        print(f"❌ Error saving statistics: {e}")
    
    # Final summary
    print(f"\n✅ ANALYSIS COMPLETE!")
    print("="*50)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets analyzed: {len(tda_analysis_results)}")
    print(f"Output files: Available in outputs/ and statistics/")
    
    return tda_analysis_results, connectome_comparison_results if len(tda_analysis_results) >= 2 else None


if __name__ == "__main__":
    # Display recommendation first
    print(recommend_distance_metric())
    print("\n" + "="*80)
    
    # Run comprehensive analysis
    hemisphere_analysis_results, distance_comparison_results = main()
    
    # Final recommendation
    print(f"\n🎯 NEXT STEPS:")
    print("="*30)
    print("1. 📊 Examine the generated CSV files in statistics/")
    print("2. 🔍 Use H1 Wasserstein distances for connectome comparison")
    print("3. 📈 Analyze persistence entropy for complexity measures")
    print("4. 🧠 Apply to your specific research questions")
    print("5. 📋 Cite the appropriate TDA methods in publications")
