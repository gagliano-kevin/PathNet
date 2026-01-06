#===================================================================================================================================
#===================================================================================================================================
#------------------------------- run this file from project root: python -m new_features_comparison.test02 -------------------------
# python -u -m new_features_comparison.test02 |& tee new_features_comparison/test02_output.txt
#===================================================================================================================================
#===================================================================================================================================

from source.PathNet2 import Trainer 

from source.california_housing_utils import get_california_housing_data
from source.test_utils import generate_statistical_summary, generate_plots, save_metrics, load_metrics

import torch.nn as nn
import numpy as np

import warnings

ITERATIONS = 2000
RUNS = 3
SAVE_TRAINED_MODEL = False
MODEL_NAME_PREFIX = "housing_model"
EARLY_STOPPING = False
E_S_PATIENCE = 200
LOSS_IMPROVEMENT_THRESHOLD = 1e-3


# =========================================================================================================================================================
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#------------------------------------------------------------------- QUANTIZATION FACTOR TEST -------------------------------------------------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# =========================================================================================================================================================



# =========================================================================================================================================================
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------- SMALL NET --------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================================================================================================================

#==========================================================================================================================================================
# test 01: with static quantization factor = 1e2
#==========================================================================================================================================================

TEST_NAME = "small_net_quantization_factor_01"
STATIC_QUANTIZATION_FACTOR = 1e2

# MLP Hyperparameters
INPUT_SIZE = 8
HIDDEN_SIZE_1 = 32
HIDDEN_SIZE_2 = 16
OUTPUT_SIZE = 1

QUANTIZATION_FACTOR = 10
PARAMETER_RANGE = (-10, 10)

# Initial Kernel and Stride Settings
WEIGHT_KERNEL = [3,3]
BIAS_KERNEL = [3]
X_STRIDE = 3
Y_STRIDE = 3
DELTA_ABS = None

# Dynamic Quantization Settings
DYNAMIC_QUANTIZATION = True
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

STATIC_ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": []
}

DYNAMIC_ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": [],
    "dynamic_quantization_iterations": [],
    "dynamic_kernel_reshaping_iterations": []
}

X_train, Y_train, X_test, Y_test = get_california_housing_data()
print(f"\nTraining Data Shape: {X_train.shape}, {Y_train.shape}")
print(f"Testing Data Shape: {X_test.shape}, {Y_test.shape}\n")

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- DYNAMIC ASTAR TRAINING ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t DYNAMIC ASTAR Training Run {run + 1} ---\n")

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
    DYNAMIC_ASTAR_METRICS["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    DYNAMIC_ASTAR_METRICS["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- STATIC ASTAR TRAINING -----------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t STATIC ASTAR Training Run {run + 1} ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, OUTPUT_SIZE),
            )


    trainer = Trainer(model=model,
                             loss_fn=nn.MSELoss(),
                             quantization_factor=STATIC_QUANTIZATION_FACTOR,
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

all_results = {
    "static": STATIC_ASTAR_METRICS,
    "dynamic": DYNAMIC_ASTAR_METRICS
}

save_metrics(all_results, TEST_NAME)

"""
loaded_results = load_metrics(TEST_NAME)
if loaded_results is not None:
    STATIC_ASTAR_METRICS = loaded_results["static"]
    DYNAMIC_ASTAR_METRICS = loaded_results["dynamic"]
"""

#==========================================================================================================================================================
# test 02: with static quantization factor = 1e2
#==========================================================================================================================================================

TEST_NAME = "small_net_quantization_factor_02"
STATIC_QUANTIZATION_FACTOR = 1e4

STATIC_ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": []
}

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- STATIC ASTAR TRAINING -----------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t STATIC ASTAR Training Run {run + 1} ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, OUTPUT_SIZE),
            )


    trainer = Trainer(model=model,
                             loss_fn=nn.MSELoss(),
                             quantization_factor=STATIC_QUANTIZATION_FACTOR,
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

all_results = {
    "static": STATIC_ASTAR_METRICS,
    "dynamic": DYNAMIC_ASTAR_METRICS
}

save_metrics(all_results, TEST_NAME)




# =========================================================================================================================================================
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------- MEDIUM NET --------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================================================================================================================

#==========================================================================================================================================================
# test 01: with static quantization factor = 1e2
#==========================================================================================================================================================

TEST_NAME = "medium_net_quantization_factor_01"
STATIC_QUANTIZATION_FACTOR = 1e2

# MLP Hyperparameters
HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 32
HIDDEN_SIZE_3 = 16

# Initial Kernel and Stride Settings
WEIGHT_KERNEL = [4,4]
BIAS_KERNEL = [4]
X_STRIDE = 4
Y_STRIDE = 4

STATIC_ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": []
}

DYNAMIC_ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": [],
    "dynamic_quantization_iterations": [],
    "dynamic_kernel_reshaping_iterations": []
}

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- DYNAMIC ASTAR TRAINING ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t DYNAMIC ASTAR Training Run {run + 1} ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_3),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_3, OUTPUT_SIZE),
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
    DYNAMIC_ASTAR_METRICS["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    DYNAMIC_ASTAR_METRICS["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- STATIC ASTAR TRAINING ----------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t STATIC ASTAR Training Run {run + 1} ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_3),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_3, OUTPUT_SIZE),
            )


    trainer = Trainer(model=model,
                             loss_fn=nn.MSELoss(),
                             quantization_factor=STATIC_QUANTIZATION_FACTOR,
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

all_results = {
    "static": STATIC_ASTAR_METRICS,
    "dynamic": DYNAMIC_ASTAR_METRICS
}

save_metrics(all_results, TEST_NAME)

#==========================================================================================================================================================
# test 02: with static quantization factor = 1e2
#==========================================================================================================================================================

TEST_NAME = "medium_net_quantization_factor_02"
STATIC_QUANTIZATION_FACTOR = 1e4

STATIC_ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": []
}

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- STATIC ASTAR TRAINING ----------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t STATIC ASTAR Training Run {run + 1} ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_3),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_3, OUTPUT_SIZE),
            )


    trainer = Trainer(model=model,
                             loss_fn=nn.MSELoss(),
                             quantization_factor=STATIC_QUANTIZATION_FACTOR,
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

all_results = {
    "static": STATIC_ASTAR_METRICS,
    "dynamic": DYNAMIC_ASTAR_METRICS
}

save_metrics(all_results, TEST_NAME)




# =========================================================================================================================================================
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------- BIG NET ----------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================================================================================================================

#==========================================================================================================================================================
# test 01: with static quantization factor = 1e2
#==========================================================================================================================================================

TEST_NAME = "big_net_quantization_factor_01"
STATIC_QUANTIZATION_FACTOR = 1e2

# MLP Hyperparameters
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64
HIDDEN_SIZE_3 = 32
HIDDEN_SIZE_4 = 16

# Initial Kernel and Stride Settings
WEIGHT_KERNEL = [5,5]
BIAS_KERNEL = [5]
X_STRIDE = 5
Y_STRIDE = 5

STATIC_ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": []
}

DYNAMIC_ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": [],
    "dynamic_quantization_iterations": [],
    "dynamic_kernel_reshaping_iterations": []
}

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- DYNAMIC ASTAR TRAINING ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t DYNAMIC ASTAR Training Run {run + 1} ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_3),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_3, HIDDEN_SIZE_4),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_4, OUTPUT_SIZE),
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
    DYNAMIC_ASTAR_METRICS["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    DYNAMIC_ASTAR_METRICS["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- STATIC ASTAR TRAINING ----------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t STATIC ASTAR Training Run {run + 1} ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_3),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_3, HIDDEN_SIZE_4),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_4, OUTPUT_SIZE),
            )


    trainer = Trainer(model=model,
                             loss_fn=nn.MSELoss(),
                             quantization_factor=STATIC_QUANTIZATION_FACTOR,
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

all_results = {
    "static": STATIC_ASTAR_METRICS,
    "dynamic": DYNAMIC_ASTAR_METRICS
}

save_metrics(all_results, TEST_NAME)

#==========================================================================================================================================================
# test 02: with static quantization factor = 1e2
#==========================================================================================================================================================

TEST_NAME = "big_net_quantization_factor_02"
STATIC_QUANTIZATION_FACTOR = 1e4

STATIC_ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": []
}

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- STATIC ASTAR TRAINING ----------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t STATIC ASTAR Training Run {run + 1} ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_3),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_3, HIDDEN_SIZE_4),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_4, OUTPUT_SIZE),
            )

    trainer = Trainer(model=model,
                             loss_fn=nn.MSELoss(),
                             quantization_factor=STATIC_QUANTIZATION_FACTOR,
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

all_results = {
    "static": STATIC_ASTAR_METRICS,
    "dynamic": DYNAMIC_ASTAR_METRICS
}

save_metrics(all_results, TEST_NAME)


#==========================================================================================================================================================
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                                                END OF QUANTIZATION FACTOR TESTS
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#==========================================================================================================================================================






# =========================================================================================================================================================
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#--------------------------------------------------------------------- DYNAMIC KERNELS TEST ---------------------------------------------------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# =========================================================================================================================================================



# =========================================================================================================================================================
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------- SMALL NET --------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================================================================================================================

#==========================================================================================================================================================
# test 01
#==========================================================================================================================================================

TEST_NAME = "small_net_dynamic_kernels_01"

# MLP Hyperparameters
INPUT_SIZE = 8
HIDDEN_SIZE_1 = 32
HIDDEN_SIZE_2 = 16
OUTPUT_SIZE = 1

QUANTIZATION_FACTOR = 10
PARAMETER_RANGE = (-10, 10)

# Initial Kernel and Stride Settings
WEIGHT_KERNEL = [4,4]
BIAS_KERNEL = [4]
X_STRIDE = 4
Y_STRIDE = 4
DELTA_ABS = None

# Dynamic Quantization Settings
DYNAMIC_QUANTIZATION = False
D_Q_PATIENCE = 100
QUANTIZATION_FACTOR_MULTIPLIER = 10
MAX_QUANTIZATION_FACTOR = 1e4

# Dynamic Kernel Reshaping Settings
DYNAMIC_KERNEL_RESHAPING = True
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

STATIC_ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": []
}

DYNAMIC_ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": [],
    "dynamic_quantization_iterations": [],
    "dynamic_kernel_reshaping_iterations": []
}

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- DYNAMIC ASTAR TRAINING ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t DYNAMIC ASTAR Training Run {run + 1} ---\n")

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
    DYNAMIC_ASTAR_METRICS["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    DYNAMIC_ASTAR_METRICS["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- STATIC ASTAR TRAINING -----------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t STATIC ASTAR Training Run {run + 1} ---\n")

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
                             weight_kernel = MIN_WEIGHT_KERNEL, bias_kernel = MIN_BIAS_KERNEL, x_stride=MIN_X_STRIDE, y_stride=MIN_Y_STRIDE, delta_abs=DELTA_ABS,
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

all_results = {
    "static": STATIC_ASTAR_METRICS,
    "dynamic": DYNAMIC_ASTAR_METRICS
}

save_metrics(all_results, TEST_NAME)

"""
loaded_results = load_metrics(TEST_NAME)
if loaded_results is not None:
    STATIC_ASTAR_METRICS = loaded_results["static"]
    DYNAMIC_ASTAR_METRICS = loaded_results["dynamic"]
"""

# =========================================================================================================================================================
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------- MEDIUM NET --------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================================================================================================================

#==========================================================================================================================================================
# test 01
#==========================================================================================================================================================

TEST_NAME = "medium_net_dynamic_kernels_01"

# MLP Hyperparameters
HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 32
HIDDEN_SIZE_3 = 16

# Initial Kernel and Stride Settings
WEIGHT_KERNEL = [6,6]
BIAS_KERNEL = [6]
X_STRIDE = 6
Y_STRIDE = 6

MIN_WEIGHT_KERNEL = [2,2]
MIN_BIAS_KERNEL = [2]
MIN_X_STRIDE = 2
MIN_Y_STRIDE = 2

STATIC_ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": []
}

DYNAMIC_ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": [],
    "dynamic_quantization_iterations": [],
    "dynamic_kernel_reshaping_iterations": []
}

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- DYNAMIC ASTAR TRAINING ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t DYNAMIC ASTAR Training Run {run + 1} ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_3),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_3, OUTPUT_SIZE),
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
    DYNAMIC_ASTAR_METRICS["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    DYNAMIC_ASTAR_METRICS["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- STATIC ASTAR TRAINING ----------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

WEIGHT_KERNEL = [4,4]
BIAS_KERNEL = [4]
X_STRIDE = 4
Y_STRIDE = 4

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t STATIC ASTAR Training Run {run + 1} ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_3),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_3, OUTPUT_SIZE),
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

all_results = {
    "static": STATIC_ASTAR_METRICS,
    "dynamic": DYNAMIC_ASTAR_METRICS
}

save_metrics(all_results, TEST_NAME)

#==========================================================================================================================================================
# test 02: with different weight kernel, bias kernel, x stride, y stride
#==========================================================================================================================================================

TEST_NAME = "medium_net_dynamic_kernels_02"

STATIC_ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": []
}

WEIGHT_KERNEL = [2,2]
BIAS_KERNEL = [2]
X_STRIDE = 2
Y_STRIDE = 2

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- STATIC ASTAR TRAINING ----------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t STATIC ASTAR Training Run {run + 1} ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_3),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_3, OUTPUT_SIZE),
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

all_results = {
    "static": STATIC_ASTAR_METRICS,
    "dynamic": DYNAMIC_ASTAR_METRICS
}

save_metrics(all_results, TEST_NAME)




# =========================================================================================================================================================
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------- BIG NET ----------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================================================================================================================

#==========================================================================================================================================================
# test 01
#==========================================================================================================================================================

TEST_NAME = "big_net_dynamic_kernels_01"

# MLP Hyperparameters
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64
HIDDEN_SIZE_3 = 32
HIDDEN_SIZE_4 = 16

# Initial Kernel and Stride Settings
WEIGHT_KERNEL = [8,8]
BIAS_KERNEL = [8]
X_STRIDE = 8
Y_STRIDE = 8

MIN_WEIGHT_KERNEL = [3,3]
MIN_BIAS_KERNEL = [3]
MIN_X_STRIDE = 3
MIN_Y_STRIDE = 3

STATIC_ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": []
}

DYNAMIC_ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": [],
    "dynamic_quantization_iterations": [],
    "dynamic_kernel_reshaping_iterations": []
}

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- DYNAMIC ASTAR TRAINING ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t DYNAMIC ASTAR Training Run {run + 1} ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_3),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_3, HIDDEN_SIZE_4),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_4, OUTPUT_SIZE),
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
    DYNAMIC_ASTAR_METRICS["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    DYNAMIC_ASTAR_METRICS["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- STATIC ASTAR TRAINING ----------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

WEIGHT_KERNEL = [6,6]
BIAS_KERNEL = [6]
X_STRIDE = 6
Y_STRIDE = 6

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t STATIC ASTAR Training Run {run + 1} ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_3),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_3, HIDDEN_SIZE_4),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_4, OUTPUT_SIZE),
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

all_results = {
    "static": STATIC_ASTAR_METRICS,
    "dynamic": DYNAMIC_ASTAR_METRICS
}

save_metrics(all_results, TEST_NAME)

#==========================================================================================================================================================
# test 02
#==========================================================================================================================================================

TEST_NAME = "big_net_dynamic_kernels_02"

STATIC_ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": []
}

WEIGHT_KERNEL = [3,3]
BIAS_KERNEL = [3]
X_STRIDE = 3
Y_STRIDE = 3

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- STATIC ASTAR TRAINING ----------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t STATIC ASTAR Training Run {run + 1} ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_3),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_3, HIDDEN_SIZE_4),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_4, OUTPUT_SIZE),
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

all_results = {
    "static": STATIC_ASTAR_METRICS,
    "dynamic": DYNAMIC_ASTAR_METRICS
}

save_metrics(all_results, TEST_NAME)