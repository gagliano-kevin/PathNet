#===================================================================================================================================
#===================================================================================================================================
#-------------- run this file from project root: python -m astar_optimization.grid_search.circle_grid_search -----------------------
#===================================================================================================================================
#===================================================================================================================================

import torch
import torch.nn as nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from source.PathNet import GridSearchTrainer

# Parameters for synthetic dataset
n_samples = 1000
noise_level = 0.1
factor_level = 0.5 
random_seed = 42

# X will have 2 features, y will have 2 classes (0 or 1)
# class 0: inner circle, class 1: outer circle
X, y = make_circles(
    n_samples=n_samples,
    noise=noise_level, 
    factor=factor_level,
    random_state=random_seed
)

# Plot the generated dataset for visualization
plt.figure(figsize=(6, 6))
# Create a scatter plot where the color 'c' is determined by the label 'y'.
# The 'coolwarm' colormap is useful for binary classification.
plt.scatter(
    X[:, 0], # Feature 1 (X-axis)
    X[:, 1], # Feature 2 (Y-axis)
    c=y, 
    cmap=plt.cm.coolwarm,
    edgecolor='k', # Black border around points
    s=40 # Size of points
)
plt.title(f"Synthetic Circles Dataset (Noise={noise_level}, Factor={factor_level})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True, linestyle='--', alpha=0.6)
#plt.show() 
plt.savefig("circles_dataset.png") 

# Stratify over y (labels) to maintain class proportions in train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_seed, stratify=y
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Info of the dataset
print("\n--- Data Preparation Summary ---")
print(f"Input Feature Count (input_size): {X.shape[1]}")        # Should be 2
print(f"Output Class Count (num_classes): {len(np.unique(y))}") # Should be 2
print(f"Training Set Size: {X_train_tensor.shape[0]}")
print(f"Test Set Size: {X_test_tensor.shape[0]}\n")

# Simple neural network model for circle classification
model = nn.Sequential(
    nn.Linear(2, 4),  
    nn.ReLU(),
    nn.Linear(4, 2)   
)

grid_search_trainer = GridSearchTrainer(
    models=[model],
    loss_funcs=[nn.CrossEntropyLoss()],
    quantization_factors=[1, 2],
    #quantization_factors=[1, 2, 5],
    parameter_ranges=[(-5, 5)],
    #parameter_ranges=[(-4, 4), (-5, 5)],
    param_fractions=[1.0],
    #param_fractions=[0.5, 1.0],
    max_iterations=[1000],
    log_freq=[500],
    target_losses=[0.0001],
    update_strategies=[2],
    #update_strategies=[0, 1, 2],
    g_ini_vals=[0],
    g_steps=[0.01],
    alphas=[0.5],
    scale_fs=[True],
    #scale_fs=[True, False],
    debug_mlps=True
)

grid_search_trainer.run_grid_search(X_train_tensor, y_train_tensor, log_filename='circle_grid_search_log.txt')
