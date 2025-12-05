"""
Example demonstrating the log_style parameter for controlling training output format.

This example shows how to use the 'log_style' parameter to control whether the model
uses notebook-style or Lightning-style progress bars during training.
"""

import scanpy as sc
import Garfield as gf

# Load example data
adata = sc.datasets.pbmc3k()

# Example 1: Auto-detect environment (default)
# - In Jupyter notebooks: uses notebook-style progress bar
# - In terminal: uses PyTorch Lightning progress bar
model = gf.Garfield({
    'adata_list': [adata],
    'profile': 'RNA',
    'n_epochs': 10,
    'log_style': 'auto',  # This is the default
})
model.train()

# Example 2: Force notebook-style progress bar
# Use this for clean, compact logging in notebooks
model_notebook = gf.Garfield({
    'adata_list': [adata],
    'profile': 'RNA',
    'n_epochs': 10,
    'log_style': 'notebook',
})
model_notebook.train()

# Example 3: Force Lightning-style progress bar
# Use this for detailed PyTorch Lightning logs (useful for debugging)
model_lightning = gf.Garfield({
    'adata_list': [adata],
    'profile': 'RNA',
    'n_epochs': 10,
    'log_style': 'lightning',
})
model_lightning.train()

# Example 4: Combined with verbose mode
# When verbose=True, all loss components are shown in the progress bar
model_verbose = gf.Garfield({
    'adata_list': [adata],
    'profile': 'RNA',
    'n_epochs': 10,
    'log_style': 'notebook',
    'verbose': True,  # Show all loss components
})
model_verbose.train()
