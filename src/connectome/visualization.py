#!/usr/bin/env python3
"""
Visualization Module
===================

Handles all plotting and visualization for tractography and TDA results.
Provides clean, publication-ready figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class TractographyVisualizer:
    """
    Creates visualizations for tractography data and analysis results.
    
    Handles streamline plots, point clouds, and persistence diagrams
    with consistent styling and clear annotations.
    """
    
    def __init__(self, style='default'):
        """
        Initialize visualizer with plotting style.
        
        Args:
            style (str): Matplotlib style to use
        """
        plt.style.use(style)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Clean color palette
        
    def plot_streamlines(self, streamlines, n_sample=100, save_path=None):
        """
        Create 2D projections and 3D view of streamlines.
        
        Args:
            streamlines (list): List of streamline arrays
            n_sample (int): Number of streamlines to plot (for performance)
            save_path (str, optional): Path to save figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Sample streamlines for visualization
        if len(streamlines) > n_sample:
            indices = np.random.choice(len(streamlines), n_sample, replace=False)
            plot_streamlines = [streamlines[i] for i in indices]
        else:
            plot_streamlines = streamlines
            
        fig = plt.figure(figsize=(12, 9))
        fig.suptitle(f'Streamlines Visualization ({len(plot_streamlines)} of {len(streamlines)} shown)', 
                     fontsize=14, fontweight='bold')
        
        # 2D Projections
        projections = [
            ('XY (Axial)', 0, 1, (2, 2, 1)),
            ('XZ (Coronal)', 0, 2, (2, 2, 2)), 
            ('YZ (Sagittal)', 1, 2, (2, 2, 3))
        ]
        
        for title, x_idx, y_idx, subplot_pos in projections:
            ax = plt.subplot(*subplot_pos)
            
            for streamline in plot_streamlines:
                ax.plot(streamline[:, x_idx], streamline[:, y_idx], 
                       alpha=0.4, linewidth=0.5, color=self.colors[0])
                       
            ax.set_title(title)
            ax.set_xlabel(f'{["X", "Y", "Z"][x_idx]} (mm)')
            ax.set_ylabel(f'{["X", "Y", "Z"][y_idx]} (mm)')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
        
        # 3D View
        ax_3d = plt.subplot(2, 2, 4, projection='3d')
        
        for streamline in plot_streamlines:
            ax_3d.plot(streamline[:, 0], streamline[:, 1], streamline[:, 2],
                      alpha=0.4, linewidth=0.5, color=self.colors[0])
                      
        ax_3d.set_title('3D View')
        ax_3d.set_xlabel('X (mm)')
        ax_3d.set_ylabel('Y (mm)')
        ax_3d.set_zlabel('Z (mm)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved streamlines plot: {save_path}")
            
        return fig
    
    def plot_point_cloud(self, points, save_path=None):
        """
        Visualize 3D point cloud with multiple projections.
        
        Args:
            points (np.ndarray): Nx3 array of 3D coordinates
            save_path (str, optional): Path to save figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig = plt.figure(figsize=(12, 9))
        fig.suptitle(f'Point Cloud ({len(points):,} points)', 
                     fontsize=14, fontweight='bold')
        
        # 2D Projections
        projections = [
            ('XY Projection', 0, 1, (2, 2, 1)),
            ('XZ Projection', 0, 2, (2, 2, 2)),
            ('YZ Projection', 1, 2, (2, 2, 3))
        ]
        
        for title, x_idx, y_idx, subplot_pos in projections:
            ax = plt.subplot(*subplot_pos)
            
            ax.scatter(points[:, x_idx], points[:, y_idx], 
                      alpha=0.6, s=1, c=self.colors[1])
                      
            ax.set_title(title)
            ax.set_xlabel(f'{["X", "Y", "Z"][x_idx]} (mm)')
            ax.set_ylabel(f'{["X", "Y", "Z"][y_idx]} (mm)')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
        
        # 3D View
        ax_3d = plt.subplot(2, 2, 4, projection='3d')
        ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2],
                     alpha=0.6, s=1, c=self.colors[1])
                     
        ax_3d.set_title('3D View')
        ax_3d.set_xlabel('X (mm)')
        ax_3d.set_ylabel('Y (mm)')
        ax_3d.set_zlabel('Z (mm)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved point cloud plot: {save_path}")
            
        return fig
    
    def plot_persistence_diagrams(self, diagrams, save_path=None):
        """
        Plot persistence diagrams for TDA results.
        
        Args:
            diagrams (list): List of persistence diagrams from ripser
            save_path (str, optional): Path to save figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        n_dims = len(diagrams)
        fig, axes = plt.subplots(1, n_dims, figsize=(5*n_dims, 5))
        
        if n_dims == 1:
            axes = [axes]
            
        fig.suptitle('Persistence Diagrams', fontsize=14, fontweight='bold')
        
        for dim, (ax, diagram) in enumerate(zip(axes, diagrams)):
            if len(diagram) == 0:
                ax.text(0.5, 0.5, 'No features', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'H{dim} (0 features)')
            else:
                births, deaths = diagram[:, 0], diagram[:, 1]
                
                # Handle infinite persistence (common in H0)
                finite_mask = np.isfinite(deaths)
                if np.any(finite_mask):
                    finite_births = births[finite_mask]
                    finite_deaths = deaths[finite_mask]
                    ax.scatter(finite_births, finite_deaths, 
                             alpha=0.7, s=20, color=self.colors[dim])
                    
                    # Diagonal line
                    max_val = max(np.max(finite_births), np.max(finite_deaths))
                    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=1)
                    
                # Show infinite persistence points on y-axis
                infinite_mask = ~finite_mask
                if np.any(infinite_mask):
                    inf_births = births[infinite_mask]
                    y_pos = ax.get_ylim()[1] * 0.95  # Near top of plot
                    ax.scatter(inf_births, [y_pos] * len(inf_births), 
                             alpha=0.7, s=20, color=self.colors[dim], 
                             marker='^', label='Infinite persistence')
                    ax.legend()
                
                ax.set_title(f'H{dim} ({len(diagram)} features)')
                
            ax.set_xlabel('Birth')
            ax.set_ylabel('Death')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved persistence diagrams: {save_path}")
            
        return fig
    
    def plot_betti_curves(self, diagrams, save_path=None):
        """
        Plot Betti curves showing number of features over filtration.
        
        Args:
            diagrams (list): List of persistence diagrams from ripser
            save_path (str, optional): Path to save figure
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Create filtration scale
        all_finite_values = []
        for diagram in diagrams:
            if len(diagram) > 0:
                finite_mask = np.isfinite(diagram).all(axis=1)
                if np.any(finite_mask):
                    all_finite_values.extend(diagram[finite_mask].flatten())
        
        if not all_finite_values:
            print("No finite persistence values found for Betti curves")
            return None
            
        t_min, t_max = min(all_finite_values), max(all_finite_values)
        t_grid = np.linspace(t_min, t_max, 200)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for dim, diagram in enumerate(diagrams):
            if len(diagram) == 0:
                continue
                
            # Count features alive at each filtration value
            births = diagram[:, 0]
            deaths = diagram[:, 1]
            
            # Only consider finite features for Betti curves
            finite_mask = np.isfinite(deaths)
            if not np.any(finite_mask):
                continue
                
            finite_births = births[finite_mask]
            finite_deaths = deaths[finite_mask]
            
            # Compute Betti curve
            betti_values = []
            for t in t_grid:
                alive = (finite_births <= t) & (t < finite_deaths)
                betti_values.append(np.sum(alive))
            
            ax.plot(t_grid, betti_values, label=f'β{dim}', 
                   linewidth=2, color=self.colors[dim])
        
        ax.set_xlabel('Filtration parameter')
        ax.set_ylabel('Number of features')
        ax.set_title('Betti Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved Betti curves: {save_path}")
            
        return fig
