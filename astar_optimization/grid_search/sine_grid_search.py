#===================================================================================================================================
#===================================================================================================================================
#-------------- run this file from project root: python -m astar_optimization.grid_search.sine_grid_search -------------------------
#===================================================================================================================================
#===================================================================================================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from source.sinusoidal_func_utils import generate_sinusoidal_tensor

from source.PathNet import GridSearchTrainer


NUM_SAMPLES = 1000
MIN_ANGLE = 0
MAX_ANGLE = 4 * np.pi
NOISE_LEVEL = 0.1
ITERATIONS = 200

RUNS = 3

LOG_FILE_ASTAR = "sine_model_astar_multiple_runs"

X_train_tensor, y_train_tensor = generate_sinusoidal_tensor(num_samples=NUM_SAMPLES, min_angle=MIN_ANGLE, max_angle=MAX_ANGLE, noise_level=NOISE_LEVEL)

model = nn.Sequential(
        nn.Linear(1, 4),  
        nn.ReLU(),
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
        nn.Tanh()
        )


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                                       INDIVIDUAL PARAMETER TESTS
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# TESTING ONLY MAX ITERATIONS PARAMETER: 1000, 5000, 10000

grid_search_trainer = GridSearchTrainer(
    models=[model],
    loss_funcs=[nn.MSELoss()],
    quantization_factors=[10],
    parameter_ranges=[(-10, 10)],
    weight_kernels = [[2,2]], 
    bias_kernels = [[2]], 
    strides=[1], 
    max_iterations=[10, 50, 100],
    log_freq=[100],
    debug_mlps=True
)

grid_search_trainer.run_grid_search(X_train_tensor, y_train_tensor, runs_per_config=RUNS, enable_training_history_logging=True, log_filename='sine_test_max_iterations')

#grid_search_trainer.plot_grid_search_trend(log_filename="sine_test_max_iterations", metric="loss_history")

grid_search_trainer.plot_avg_loss(file_name="sine_test_max_iterations")


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#TESTING ONLY QUANTIZATION FACTOR PARAMETER: 1, 10, 100

grid_search_trainer = GridSearchTrainer(
    models=[model],
    loss_funcs=[nn.MSELoss()],
    quantization_factors=[1, 10, 100],
    parameter_ranges=[(-10, 10)],
    weight_kernels = [[2,2]], 
    bias_kernels = [[2]], 
    strides=[1], 
    max_iterations=[ITERATIONS],
    log_freq=[100],
    debug_mlps=True
)

grid_search_trainer.run_grid_search(X_train_tensor, y_train_tensor, runs_per_config=2, enable_training_history_logging=True, log_filename='sine_test_quantization_factor')

#grid_search_trainer.plot_grid_search_trend(log_filename="sine_test_quantization_factor", metric="loss_history")

grid_search_trainer.plot_avg_loss(file_name="sine_test_quantization_factor")


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#TESTING ONLY PARAMETER RANGE PARAMETER: -5 to 5, -10 to 10 and -20 to 20

grid_search_trainer = GridSearchTrainer(
    models=[model],
    loss_funcs=[nn.MSELoss()],
    quantization_factors=[10],
    parameter_ranges=[(-5, 5), (-10, 10), (-20, 20)],
    weight_kernels = [[2,2]], 
    bias_kernels = [[2]], 
    strides=[1], 
    max_iterations=[ITERATIONS],
    log_freq=[100],
    debug_mlps=True
)

grid_search_trainer.run_grid_search(X_train_tensor, y_train_tensor, runs_per_config=2, enable_training_history_logging=True, log_filename='sine_test_parameter_range')

#grid_search_trainer.plot_grid_search_trend(log_filename="sine_test_parameter_range", metric="loss_history")

grid_search_trainer.plot_avg_loss(file_name="sine_test_parameter_range")


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#TESTING ONLY WEIGHT KERNEL SIZE 


grid_search_trainer = GridSearchTrainer(
    models=[model],
    loss_funcs=[nn.MSELoss()],
    quantization_factors=[10],
    parameter_ranges=[(-10, 10)],
    weight_kernels = [[2,2], [3,3]], 
    bias_kernels = [[2],[3]], 
    strides=[1,2], 
    max_iterations=[ITERATIONS],
    log_freq=[100],
    debug_mlps=True
)

grid_search_trainer.run_grid_search(X_train_tensor, y_train_tensor, runs_per_config=RUNS, enable_training_history_logging=True, log_filename='sine_test_kernel_stride')

#grid_search_trainer.plot_grid_search_trend(log_filename="sine_test_kernel_stride", metric="loss_history")

grid_search_trainer.plot_avg_loss(file_name="sine_test_kernel_stride")


