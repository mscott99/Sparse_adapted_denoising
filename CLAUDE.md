# Sparse Adapted Denoising Codebase Guide

## Build & Test Commands
- Run tests: `pytest tests/`
- Run single test: `pytest tests/test_file.py::test_function`
- Run main experiment: `python run_local.py`

## Code Style Guidelines

### Imports
- Group imports: system imports first, then third-party libraries (numpy, scipy), then local modules
- Example order: `numpy`, `scipy`, `pandas`, then project imports

### Type Annotations
- Use type hints consistently: `NDArray`, `Tuple`, `List`, etc.
- Import from `typing` and `numpy.typing`

### Docstrings
- Use NumPy-style docstrings with Parameters and Returns sections
- Document parameters with types and descriptions

### Naming Conventions
- Functions: `snake_case`
- Variables: descriptive, lowercase with underscores
- Type hints: `PascalCase`

### Error Handling
- Use assertions for conditions that must be true
- Raise exceptions with clear error messages when appropriate

## Server Deployment Structure
- This project is designed to run on Slurm using framework at: `/Users/matthewscott/Prog/Experiments_on_Slurm`
- Each experiment module must export two key functions:
  1. `make_df()`: Returns DataFrame where each row = one experiment configuration
  2. `experiment(row)`: Takes a row from make_df() and returns dict with results
- Helper scripts for server deployment: `/Users/matthewscott/Prog/server_scripts`

## Key Functionality
- Project is split into modular components:
  - `sparse_recov/sampling_recov_functions.py`: Compressed sensing operations - `SF2_func`, `SF2_transpose_map`, `soft_threshold`, etc.
  - `sparse_recov/data_analysis.py`: Statistical plotting functions - `perform_least_squares_fit`, `add_fit_lines_to_plot`, `plot_on_facets`
  - `sparse_recov/uneven_sparse_experiment.py`: Main experiment implementation
- Visualization and data analysis via Jupyter notebooks in `plots/` directory