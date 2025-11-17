#===================================================================================================================================
#===================================================================================================================================
#-------------- run this file from project root: python -m astar_optimization.grid_search.iris_grid_search ------------------------
#===================================================================================================================================
#===================================================================================================================================

import torch
import torch.nn as nn
from source.iris_utils import get_iris_data_tensors, print_iris_data_info
from source.PathNet import GridSearchTrainer

print_iris_data_info()

X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = get_iris_data_tensors()

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
    parameter_ranges=[(-5, 5)],
    param_fractions=[1.0],
    max_iterations=[1000],
    log_freq=[500],
    target_losses=[0.0001]
)

grid_search_trainer.run_grid_search(X_train_tensor, y_train_tensor, runs_per_config=2, enable_training_history_logging=False, log_filename='iris_grid_search_log.txt')