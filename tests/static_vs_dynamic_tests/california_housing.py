#===============================================================================================================================================
#===============================================================================================================================================
#------ python -u -m tests.static_vs_dynamic_tests.california_housing |& tee tests/static_vs_dynamic_tests/california_housing_output.txt -------
#===============================================================================================================================================
#===============================================================================================================================================
import torch.nn as nn

from source.PathNet import Trainer
from source.utils.dataset_utils.housing_utils import get_california_housing_data

from source.utils.plot_utils import generate_plots, generate_statistical_summary, format_sci,  save_metrics, load_metrics

ITERATIONS = 50
RUNS = 1
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

TEST_NAME = "small_net_quantization_factor"
DATASET_NAME = "California Housing"

# MLP Hyperparameters
INPUT_SIZE = 8
HIDDEN_SIZE_1 = 32
HIDDEN_SIZE_2 = 16
OUTPUT_SIZE = 1

QUANTIZATION_FACTOR = 10
STATIC_QUANTIZATION_FACTOR_1 = 1e2
STATIC_QUANTIZATION_FACTOR_2 = 1e4

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


DYNAMIC_ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": [],
    "dynamic_quantization_iterations": [],
    "dynamic_kernel_reshaping_iterations": []
}

STATIC_1_ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": []
}

STATIC_2_ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": []
}

X_train, Y_train, X_val, Y_val, X_test, Y_test = get_california_housing_data()
print(f"\nTraining Data Shape: {X_train.shape}, {Y_train.shape}")
print(f"Validation Data Shape: {X_val.shape}, {Y_val.shape}")
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
                             quantization_factor=STATIC_QUANTIZATION_FACTOR_1,
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

    STATIC_1_ASTAR_METRICS["losses"].append(trainer.loss_history)
    STATIC_1_ASTAR_METRICS["training_times"].append(trainer.training_time)
    STATIC_1_ASTAR_METRICS["final_losses"].append(trainer.best_node.h_val)



#==========================================================================================================================================================
# test 01 bis: with static quantization factor = 1e2
#==========================================================================================================================================================

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
                             quantization_factor=STATIC_QUANTIZATION_FACTOR_2,
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

    STATIC_2_ASTAR_METRICS["losses"].append(trainer.loss_history)
    STATIC_2_ASTAR_METRICS["training_times"].append(trainer.training_time)
    STATIC_2_ASTAR_METRICS["final_losses"].append(trainer.best_node.h_val)

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

metric_list = [DYNAMIC_ASTAR_METRICS, STATIC_1_ASTAR_METRICS, STATIC_2_ASTAR_METRICS]

labels = [
    "Dynamic A-star", 
    f"Static A-star QF={format_sci(STATIC_QUANTIZATION_FACTOR_1)}", 
    f"Static A-star QF={format_sci(STATIC_QUANTIZATION_FACTOR_2)}"
]
generate_statistical_summary(metric_list,labels, TEST_NAME)

generate_plots(metric_list, labels, TEST_NAME, DATASET_NAME)

all_results = {
    "static_1": STATIC_1_ASTAR_METRICS,
    "static_2": STATIC_2_ASTAR_METRICS,
    "dynamic": DYNAMIC_ASTAR_METRICS
}

save_metrics(all_results, TEST_NAME)
