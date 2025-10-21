#===================================================================================================================================
#===================================================================================================================================
#-------------- run this file from project root: python -m astar_optimization.grid_search.iris_grid_search ------------------------
#===================================================================================================================================
#===================================================================================================================================

import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from source.PathNet import GridSearchTrainer

iris = load_iris()
X, y = iris.data, iris.target

# Scaling features for better performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Info of the dataset
print("--- Data Variables Ready for Custom Model ---")
print(f"Input Feature Count (input_size): {X_train.shape[1]}")
print(f"Output Class Count (num_classes): {len(np.unique(y))}")
print(f"Training Set Size: {X_train_tensor.shape[0]}")
print(f"Test Set Size: {X_test_tensor.shape[0]}\n")
print(f"X_train_tensor shape: {X_train_tensor.shape}\n")
print(f"y_train_tensor shape: {y_train_tensor.shape}\n")
print(f"X_test_tensor shape: {X_test_tensor.shape}\n")
print(f"y_test_tensor shape: {y_test_tensor.shape}\n")

# Simple neural network model for iris classification
model = nn.Sequential(
nn.Linear(4, 4),
nn.ReLU(),
nn.Linear(4, 3),
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

grid_search_trainer.run_grid_search(X_train_tensor, y_train_tensor, runs_per_config=2, enable_training_history_logging=False, log_filename='iris_grid_search_log.txt')