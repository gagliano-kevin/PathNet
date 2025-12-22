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

# Parameters for sine wave data generation
NUM_SAMPLES = 1000
MIN_ANGLE = 0
MAX_ANGLE = 4 * np.pi
NOISE_LEVEL = 0.1

# Numer of repeated runs for each configuration
RUNS = 10

# Default number of iterations for training
DEFAULT_ITERATIONS = 2000

# Default parameters for grid search
DEFAULT_WEIGHT_KERNEL = [3, 3]
DEFAULT_BIAS_KERNEL = [3]
DEFAULT_STRIDE = 2

DEFAULT_PARAMETER_RANGE = (-20, 20)
DEFAULT_QUANTIZATION_FACTOR = 10


LOG_FILE_ASTAR = "sine_model_astar_multiple_runs"


X_train_tensor, y_train_tensor = generate_sinusoidal_tensor(num_samples=NUM_SAMPLES, min_angle=MIN_ANGLE, max_angle=MAX_ANGLE, noise_level=NOISE_LEVEL)

"""
model = nn.Sequential(
        nn.Linear(1, 4),  
        nn.ReLU(),
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
        nn.Tanh()
        )
"""

model = nn.Sequential(
        nn.Linear(1, 8),  
        nn.ReLU(),
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 8),  
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Tanh()
        )

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                                       INDIVIDUAL PARAMETER TESTS
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# TESTING ONLY MAX ITERATIONS PARAMETER: 1000, 2000
"""
grid_search_trainer = GridSearchTrainer(
    models=[model],
    loss_funcs=[nn.MSELoss()],
    quantization_factors=[DEFAULT_QUANTIZATION_FACTOR],
    parameter_ranges=[DEFAULT_PARAMETER_RANGE],
    weight_kernels = [DEFAULT_WEIGHT_KERNEL], 
    bias_kernels = [DEFAULT_BIAS_KERNEL], 
    strides=[DEFAULT_STRIDE], 
    max_iterations=[1000, 2000],
    log_freq=[100],
    debug_mlps=True
)

grid_search_trainer.run_grid_search(X_train_tensor, y_train_tensor, runs_per_config=RUNS, enable_training_history_logging=True, log_filename='sine_test_max_iterations')

#grid_search_trainer.plot_grid_search_trend(log_filename="sine_test_max_iterations", metric="loss_history")

grid_search_trainer.plot_avg_loss(file_name="sine_test_max_iterations", parameter_name="max_iterations")

# if all parameters are to be plotted in boxplot labels 
#grid_search_trainer.plot_final_loss_boxplot(file_name="sine_test_max_iterations")

# if only the tested parameter is to be plotted in boxplot labels
grid_search_trainer.plot_final_loss_boxplot(file_name="sine_test_max_iterations", x_label="max_iterations")

grid_search_trainer.generate_final_loss_summary(file_name="sine_test_max_iterations")
"""


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#TESTING ONLY QUANTIZATION FACTOR PARAMETER: 10, 100, 1000
"""
grid_search_trainer = GridSearchTrainer(
    models=[model],
    loss_funcs=[nn.MSELoss()],
    quantization_factors=[10, 100, 1000],
    parameter_ranges=[DEFAULT_PARAMETER_RANGE],
    weight_kernels = [DEFAULT_WEIGHT_KERNEL], 
    bias_kernels = [DEFAULT_BIAS_KERNEL], 
    strides=[DEFAULT_STRIDE], 
    max_iterations=[DEFAULT_ITERATIONS],
    log_freq=[100],
    debug_mlps=True
)

grid_search_trainer.run_grid_search(X_train_tensor, y_train_tensor, runs_per_config=RUNS, enable_training_history_logging=True, log_filename='sine_test_quantization_factor')

#grid_search_trainer.plot_grid_search_trend(log_filename="sine_test_quantization_factor", metric="loss_history")

grid_search_trainer.plot_avg_loss(file_name="sine_test_quantization_factor", parameter_name="quantization_factor")

grid_search_trainer.plot_final_loss_boxplot(file_name="sine_test_quantization_factor", x_label="quantization_factor")

grid_search_trainer.generate_final_loss_summary(file_name="sine_test_quantization_factor")
"""

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#TESTING ONLY PARAMETER RANGE PARAMETER: -5 to 5, -10 to 10 and -20 to 20

#"""
grid_search_trainer = GridSearchTrainer(
    models=[model],
    loss_funcs=[nn.MSELoss()],
    quantization_factors=[DEFAULT_QUANTIZATION_FACTOR],
    parameter_ranges=[(-5, 5), (-10, 10), (-20, 20)],
    weight_kernels = [DEFAULT_WEIGHT_KERNEL], 
    bias_kernels = [DEFAULT_BIAS_KERNEL], 
    strides=[DEFAULT_STRIDE], 
    max_iterations=[DEFAULT_ITERATIONS],
    log_freq=[100],
    debug_mlps=True
)

grid_search_trainer.run_grid_search(X_train_tensor, y_train_tensor, runs_per_config=RUNS, enable_training_history_logging=True, log_filename='sine_test_parameter_range')

#grid_search_trainer.plot_grid_search_trend(log_filename="sine_test_parameter_range", metric="loss_history")

grid_search_trainer.plot_avg_loss(file_name="sine_test_parameter_range", parameter_name="parameter_range")

grid_search_trainer.plot_final_loss_boxplot(file_name="sine_test_parameter_range", x_label="parameter_range")

grid_search_trainer.generate_final_loss_summary(file_name="sine_test_parameter_range")
#"""

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#TESTING ONLY WEIGHT KERNEL SIZE 

"""
grid_search_trainer = GridSearchTrainer(
    models=[model],
    loss_funcs=[nn.MSELoss()],
    quantization_factors=[DEFAULT_QUANTIZATION_FACTOR],
    parameter_ranges=[DEFAULT_PARAMETER_RANGE],
    #weight_kernels = [[1,1], [2,2], [3,3]], 
    weight_kernels=[[2,2],[3,3]],
    #bias_kernels = [[1], [2], [3]], 
    bias_kernels=[[2],[3]],
    strides=[2], 
    max_iterations=[DEFAULT_ITERATIONS],
    log_freq=[100],
    debug_mlps=True
)

grid_search_trainer.run_grid_search(X_train_tensor, y_train_tensor, runs_per_config=RUNS, enable_training_history_logging=True, log_filename='sine_test_kernel_stride')

#grid_search_trainer.plot_grid_search_trend(log_filename="sine_test_kernel_stride", metric="loss_history")

grid_search_trainer.plot_avg_loss(file_name="sine_test_kernel_stride", parameter_name=["weight_kernel", "bias_kernel", "stride"])

grid_search_trainer.plot_final_loss_boxplot(file_name="sine_test_kernel_stride", x_label=["weight_kernel", "bias_kernel", "stride"])

grid_search_trainer.generate_final_loss_summary(file_name="sine_test_kernel_stride")
"""