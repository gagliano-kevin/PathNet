#===================================================================================================================================
#===================================================================================================================================
#------------------------------- run this file from project root: python -m new_features_comparison.test01 -------------------------
#===================================================================================================================================
#===================================================================================================================================

from source.PathNet2 import Trainer 

from source.california_housing_utils import get_california_housing_data
from source.general_utils import plot_final_loss_distribution, plot_mean_loss_with_std, pad_to_max
from source.test_utils import generate_statistical_summary, generate_plots

import torch.nn as nn
import numpy as np

import warnings


TEST_NAME = "housing_ASTAR_training_comparison"

ITERATIONS = 2000

RUNS = 1

# MLP Hyperparameters
INPUT_SIZE = 8
HIDDEN_SIZE_1 = 32
HIDDEN_SIZE_2 = 32
OUTPUT_SIZE = 1

QUANTIZATION_FACTOR = 10
PARAMETER_RANGE = (-10, 10)

# Initial Kernel and Stride Settings
WEIGHT_KERNEL = [4,4]
BIAS_KERNEL = [4]
X_STRIDE = 4
Y_STRIDE = 4
DELTA_ABS = None

EARLY_STOPPING = False
E_S_PATIENCE = 200

# Dynamic Quantization Settings
DYNAMIC_QUANTIZATION = False
D_Q_PATIENCE = 100
QUANTIZATION_FACTOR_MULTIPLIER = 10
MAX_QUANTIZATION_FACTOR = 1e4

# Dynamic Kernel Reshaping Settings
DYNAMIC_KERNEL_RESHAPING = False
D_K_R_PATIENCE = 100
X_WEIGHT_KERNEL_DECR = 1
Y_WEIGHT_KERNEL_DECR = 1
Y_BIAS_KERNEL_DECR = 1
MIN_WEIGHT_KERNEL = [2,2]
MIN_BIAS_KERNEL = [2]
X_STRIDE_DECR = 1
Y_STRIDE_DECR = 1
MIN_X_STRIDE = 2
MIN_Y_STRIDE = 2

LOSS_IMPROVEMENT_THRESHOLD = 1e-3

SAVE_TRAINED_MODEL = False
MODEL_NAME_PREFIX = "housing_model"


STATIC_ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": []
}

DYNAMIC_ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": []
}

"""
LOG_FILE_STATIC_ASTAR = "housing_model_static_astar_multiple_runs"
LOG_FILE_DYNAMIC_ASTAR = "housing_model_dynamic_astar_multiple_runs"
"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- DYNAMIC ASTAR TRAINING -----------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

X_train, Y_train, X_test, Y_test = get_california_housing_data()

print(f"\nTraining Data Shape: {X_train.shape}, {Y_train.shape}")
print(f"Testing Data Shape: {X_test.shape}, {Y_test.shape}\n")

for run in range(RUNS):
    print(f"\n--- ASTAR Training Run {run + 1} ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, OUTPUT_SIZE),
            )


    trainer = Trainer(model=model,
                             loss_fn=nn.MSELoss(),
                             quantization_factor=QUANTIZATION_FACTOR,
                             parameter_range=PARAMETER_RANGE,
                             debug_mlp=False,
                             #----------------------------------------------------------------------------------
                             weight_kernel = WEIGHT_KERNEL, bias_kernel = BIAS_KERNEL, x_stride=X_STRIDE, y_stride=Y_STRIDE, delta_abs=DELTA_ABS,
                             #----------------------------------------------------------------------------------
                             early_stopping=EARLY_STOPPING, e_s_patience=E_S_PATIENCE,
                             #----------------------------------------------------------------------------------
                             dynamic_quantization=DYNAMIC_QUANTIZATION, d_q_patience=D_Q_PATIENCE, 
                             quantization_factor_multiplier=QUANTIZATION_FACTOR_MULTIPLIER, max_quantization_factor=MAX_QUANTIZATION_FACTOR,
                             #-----------------------------------------------------------------------------------
                             dynamic_kernel_reshaping=DYNAMIC_KERNEL_RESHAPING, d_k_r_patience=D_K_R_PATIENCE, 
                             x_weight_kernel_decr=X_WEIGHT_KERNEL_DECR, y_weight_kernel_decr=Y_WEIGHT_KERNEL_DECR, y_bias_kernel_decr=Y_BIAS_KERNEL_DECR, 
                             min_weight_kernel=MIN_WEIGHT_KERNEL, min_bias_kernel=MIN_BIAS_KERNEL,
                             x_stride_decr=X_STRIDE_DECR, y_stride_decr=Y_STRIDE_DECR, min_x_stride=MIN_X_STRIDE, min_y_stride=MIN_Y_STRIDE,
                             #----------------------------------------------------------------------------------
                             loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                             #----------------------------------------------------------------------------------
                             max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_dynamic_astar_run_{run + 1}'
                             )

    trainer.train(X_train, Y_train)

    DYNAMIC_ASTAR_METRICS["losses"].append(trainer.loss_history)
    DYNAMIC_ASTAR_METRICS["training_times"].append(trainer.training_time)
    DYNAMIC_ASTAR_METRICS["final_losses"].append(trainer.best_node.h_val)


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- STATIC ASTAR TRAINING -----------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

X_train, Y_train, X_test, Y_test = get_california_housing_data()

print(f"\nTraining Data Shape: {X_train.shape}, {Y_train.shape}")
print(f"Testing Data Shape: {X_test.shape}, {Y_test.shape}\n")

for run in range(RUNS):
    print(f"\n--- ASTAR Training Run {run + 1} ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, OUTPUT_SIZE),
            )


    trainer = Trainer(model=model,
                             loss_fn=nn.MSELoss(),
                             quantization_factor=QUANTIZATION_FACTOR,
                             parameter_range=PARAMETER_RANGE,
                             debug_mlp=False,
                             #----------------------------------------------------------------------------------
                             weight_kernel = WEIGHT_KERNEL, bias_kernel = BIAS_KERNEL, x_stride=X_STRIDE, y_stride=Y_STRIDE, delta_abs=DELTA_ABS,
                             #----------------------------------------------------------------------------------
                             early_stopping=False, e_s_patience=E_S_PATIENCE,
                             #----------------------------------------------------------------------------------
                             dynamic_quantization=False, d_q_patience=D_Q_PATIENCE, 
                             quantization_factor_multiplier=QUANTIZATION_FACTOR_MULTIPLIER, max_quantization_factor=MAX_QUANTIZATION_FACTOR,
                             #-----------------------------------------------------------------------------------
                             dynamic_kernel_reshaping=False, d_k_r_patience=D_K_R_PATIENCE, 
                             x_weight_kernel_decr=X_WEIGHT_KERNEL_DECR, y_weight_kernel_decr=Y_WEIGHT_KERNEL_DECR, y_bias_kernel_decr=Y_BIAS_KERNEL_DECR, 
                             min_weight_kernel=MIN_WEIGHT_KERNEL, min_bias_kernel=MIN_BIAS_KERNEL,
                             x_stride_decr=X_STRIDE_DECR, y_stride_decr=Y_STRIDE_DECR, min_x_stride=MIN_X_STRIDE, min_y_stride=MIN_Y_STRIDE,
                             #----------------------------------------------------------------------------------
                             loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                             #----------------------------------------------------------------------------------
                             max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_static_astar_run_{run + 1}'
                             )

    trainer.train(X_train, Y_train)

    STATIC_ASTAR_METRICS["losses"].append(trainer.loss_history)
    STATIC_ASTAR_METRICS["training_times"].append(trainer.training_time)
    STATIC_ASTAR_METRICS["final_losses"].append(trainer.best_node.h_val)


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

generate_statistical_summary(STATIC_ASTAR_METRICS, DYNAMIC_ASTAR_METRICS, TEST_NAME)

generate_plots(STATIC_ASTAR_METRICS, DYNAMIC_ASTAR_METRICS, TEST_NAME)
