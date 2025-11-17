#===================================================================================================================================
#===================================================================================================================================
#-------------- run this file from project root: python -m astar_optimization.grid_search.xor_grid_search -----------------------
#===================================================================================================================================
#===================================================================================================================================

import torch
import torch.nn as nn
from source.PathNet import GridSearchTrainer

"""
    Simple MLP model for XOR problem
"""
XOR_MODEL = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

XOR_DATASET = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
XOR_LABELS = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

XOR_LOSS_FN = nn.BCELoss()

model = nn.Sequential(
nn.Linear(2, 2),
nn.ReLU(),
nn.Linear(2, 1),
nn.Sigmoid())

grid_search_trainer = GridSearchTrainer(
    models=[model],
    loss_funcs=[XOR_LOSS_FN],
    quantization_factors=[1, 2],
    parameter_ranges=[(-5, 5)],
    param_fractions=[1.0],
    max_iterations=[1000],
    log_freq=[500],
    target_losses=[0.0001],
)

grid_search_trainer.run_grid_search(XOR_DATASET, XOR_LABELS, runs_per_config=2, enable_training_history_logging=False, log_filename='xor_grid_search_log.txt')