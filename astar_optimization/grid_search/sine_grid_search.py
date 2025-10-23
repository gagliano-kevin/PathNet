#===================================================================================================================================
#===================================================================================================================================
#-------------- run this file from project root: python -m astar_optimization.grid_search.sine_grid_search ------------------------
#===================================================================================================================================
#===================================================================================================================================

import torch
import torch.nn as nn
import numpy as np
from source.sinusoidal_func_utils import generate_sinusoidal_tensor, plot_sine_predictions
from source.PathNet import GridSearchTrainer


NUM_SAMPLES = 1000
MIN_ANGLE = 0
MAX_ANGLE = 4 * np.pi
NOISE_LEVEL = 0.1
ITERATIONS = 100


X_train_tensor, y_train_tensor = generate_sinusoidal_tensor(func=torch.sin, num_samples=NUM_SAMPLES, min_angle=MIN_ANGLE, max_angle=MAX_ANGLE, noise_level=NOISE_LEVEL)


model = nn.Sequential(
    nn.Linear(1, 6),  
    nn.ReLU(),
    #nn.Tanh(),
    nn.Linear(6, 6),
    nn.ReLU(),
    #nn.Tanh(),
    nn.Linear(6, 1),
    #nn.Tanh()
    )

grid_search_trainer = GridSearchTrainer(
    models=[model],
    loss_funcs=[nn.MSELoss()],
    quantization_factors=[1, 2],
    #quantization_factors=[1, 2, 5],
    parameter_ranges=[(-5, 5)],
    #parameter_ranges=[(-4, 4), (-5, 5)],
    param_fractions=[1.0],
    #param_fractions=[0.5, 1.0],
    max_iterations=[100],
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

grid_search_trainer.run_grid_search(X_train_tensor, y_train_tensor, runs_per_config=2, enable_training_history_logging=False, log_filename='sine_grid_search_log.txt')