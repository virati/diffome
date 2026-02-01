# Refactoring Documentation

## Overview
This document describes the refactoring performed to clean up the repository and separate concerns into reusable classes.

## Changes Made

### 1. Created Core Interfaces (`src/diffome/core/`)
- **`interfaces.py`**: Added abstract base classes for key components:
  - `PointCloud`: Interface for point cloud data operations
  - `StreamlineLoader`: Interface for loading streamline data
  - `Renderer`: Interface for rendering visualizations

### 2. Separated Data Loading (`src/diffome/connectome/data_loader.py`)
Extracted data loading and processing logic from `Connectome` class:
- **`TractogramLoader`**: Concrete implementation for loading tractogram files using DIPY
- **`SyntheticStreamlineGenerator`**: Generates synthetic streamline data for testing
- **`PointCloudExtractor`**: Extracts point cloud data from streamlines

### 3. Refactored Connectome Class (`src/diffome/connectome/base.py`)
- Now implements `PointCloud` interface
- Uses dependency injection for loading (uses `TractogramLoader`)
- Separated concerns:
  - Data loading → delegated to loaders
  - Point cloud extraction → delegated to `PointCloudExtractor`
  - Rendering → accepts renderer as parameter (not hard-coded)
- Added proper docstrings and type hints

### 4. Separated TDA Computation (`src/diffome/tda/persistence.py`)
Created specialized classes for persistence computation:
- **`PersistenceComputer`**: Computes Rips complex and persistence diagrams
- **`PersistenceDiagramPlotter`**: Handles visualization of persistence diagrams

### 5. Refactored BarCode Analysis (`src/diffome/tda/barcode.py`)
- Separated computation from visualization
- Uses composition with `PersistenceComputer` and `PersistenceDiagramPlotter`
- Cleaner interface: `calculate()` method with optional plotting
- Improved error handling and documentation

### 6. Implemented Visualization Module (`src/diffome/viz/`)
- **`StreamlineRenderer`**: Renderer for streamline visualizations
- **`PersistenceDiagramRenderer`**: Renderer for persistence diagrams
- **`create_streamline_renderer()`**: Factory function for creating renderers
- All implement the `Renderer` interface

### 7. Updated Module Exports
Updated `__init__.py` files to properly export new classes:
- `src/diffome/__init__.py`: Main package exports
- `src/diffome/connectome/__init__.py`: Connectome module exports
- `src/diffome/tda/__init__.py`: TDA module exports
- `src/diffome/core/__init__.py`: Core interfaces exports

## Benefits

### Separation of Concerns
- Data loading is separate from domain logic
- Computation is separate from visualization
- Each class has a single, well-defined responsibility

### Testability
- Each component can be tested independently
- Mock implementations can be easily created using interfaces
- No more hard-coded dependencies

### Extensibility
- New loaders can be added by implementing `StreamlineLoader`
- New renderers can be added by implementing `Renderer`
- New TDA methods can extend `TDAAnalysis`

### Maintainability
- Smaller, focused classes are easier to understand
- Changes to one component don't affect others
- Clear interfaces make the codebase more approachable

## Backward Compatibility

The refactoring maintains backward compatibility with existing code:
- Original imports still work: `from diffome.connectome.base import Connectome`
- Original class interfaces are preserved
- The script in `scripts/subatlas_tda.py` should continue to work without changes

## Architecture Summary

```
diffome/
├── core/           # Abstract interfaces
│   └── interfaces.py
├── connectome/     # Connectome management
│   ├── base.py     # Main Connectome class
│   └── data_loader.py  # Loading utilities
├── tda/            # Topological data analysis
│   ├── barcode.py  # Barcode analysis
│   └── persistence.py  # Computation utilities
└── viz/            # Visualization
    └── __init__.py # Renderer implementations
```

## Future Improvements

1. Add proper unit tests for each component
2. Implement additional TDA methods (e.g., persistence landscapes)
3. Add more sophisticated visualization options
4. Create configuration management for parameters
5. Add data validation and error handling
6. Implement caching for expensive computations
