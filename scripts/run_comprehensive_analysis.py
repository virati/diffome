#!/usr/bin/env python3
"""
Comprehensive Analysis Script

This script orchestrates the complete connectome analysis pipeline.
Run from the project root directory.
"""

import sys
import os
# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
"""
Comprehensive Analysis Script
============================

Runs complete analysis pipeline including individual analyses and joint analysis
with comprehensive statistics collection and comparison.
"""

import os
import subprocess
import pandas as pd
from datetime import datetime


def run_command(cmd):
    """Run a command and return success status."""
    print(f"\n🔨 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Success")
        return True
    else:
        print(f"❌ Failed: {result.stderr}")
        return False


def main():
    """Run comprehensive analysis pipeline."""
    print("=" * 80)
    print("COMPREHENSIVE TRACTOGRAPHY TOPOLOGICAL ANALYSIS")
    print("=" * 80)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Individual analyses with new naming
    print(f"\n🧠 STEP 1: Individual hemisphere analyses")
    
    # Left hemisphere
    success1 = run_command(
        'python main.py --file "data/Petersen PD Top Left.trk" --output-prefix "petersen_left_hemisphere"'
    )
    
    # Right hemisphere  
    success2 = run_command(
        'python main.py --file "data/Peterson PD Top Right.trk" --output-prefix "petersen_right_hemisphere"'
    )
    
    # Step 2: Joint analysis
    print(f"\n🔗 STEP 2: Joint hemisphere analysis")
    success3 = run_command(
        'python main.py --joint-analysis'
    )
    
    # Step 3: Sampling robustness experiment (optional, shorter version)
    print(f"\n🎲 STEP 3: Sampling robustness experiment")
    success4 = run_command(
        'python main.py --sampling-experiment --sampling-k 5 10 --experiment-name "pd_robustness_study"'
    )
    
    # Step 4: Summary and analysis
    print(f"\n📊 STEP 4: Analysis summary")
    
    print(f"\n📋 Results Summary:")
    print(f"✅ Individual analyses: {success1 and success2}")
    print(f"✅ Joint analysis: {success3}")
    print(f"✅ Sampling experiment: {success4}")
    
    # Check outputs
    output_files = [
        'outputs/pd_top_left_persistence_diagrams.png',
        'outputs/pd_top_right_persistence_diagrams.png', 
        'outputs/pd_joint_persistence_diagrams.png'
    ]
    
    print(f"\n📁 Generated output files:")
    for file_path in output_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (missing)")
    
    # Statistics files
    stats_files = [f for f in os.listdir('statistics') if f.endswith('.csv')]
    print(f"\n📊 Statistics files ({len(stats_files)} total):")
    for file_name in sorted(stats_files)[-6:]:  # Show last 6 files
        print(f"  📄 {file_name}")
    
    # Comparative analysis
    print(f"\n🔍 COMPARATIVE ANALYSIS:")
    
    if os.path.exists('statistics') and len(stats_files) > 0:
        # Find latest TDA statistics file
        tda_files = [f for f in stats_files if 'tda.csv' in f]
        if tda_files:
            latest_tda = sorted(tda_files)[-1]
            tda_path = os.path.join('statistics', latest_tda)
            
            try:
                df = pd.read_csv(tda_path)
                print(f"\n📈 TDA Comparison (from {latest_tda}):")
                print(f"{'Dataset':<20} {'H0 Features':<12} {'H1 Features':<12} {'H1 Max Persist':<15}")
                print(f"{'-'*60}")
                
                for _, row in df.iterrows():
                    dataset = row['dataset_name']
                    h0_features = row['h0_n_features']
                    h1_features = row['h1_n_features'] 
                    h1_max = row['h1_max_persistence']
                    print(f"{dataset:<20} {h0_features:<12} {h1_features:<12} {h1_max:<15.3f}")
                    
            except Exception as e:
                print(f"Could not load statistics: {e}")
    
    print(f"\n📚 INTERPRETATION GUIDE:")
    print(f"• pd_top_left: Left hemisphere analysis")
    print(f"• pd_top_right: Right hemisphere analysis")
    print(f"• pd_joint: Combined bilateral analysis")
    print(f"• H0 features: Connected components (brain regions)")
    print(f"• H1 features: Topological loops/holes")
    print(f"• Higher H1 features → More complex topology")
    print(f"• Higher persistence → More significant features")
    
    print(f"\n🎯 RESEARCH INSIGHTS:")
    print(f"1. Compare H1 features between hemispheres for asymmetry")
    print(f"2. Joint analysis reveals bilateral topological structure")
    print(f"3. Persistence values indicate feature significance")
    print(f"4. Sampling experiments validate robustness")
    
    print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return success1 and success2 and success3


if __name__ == "__main__":
    main()
