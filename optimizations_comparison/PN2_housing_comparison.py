#===================================================================================================================================
#===================================================================================================================================
#----------------- run this file from project root: python -m optimizations_comparison.PN2_housing_comparison ----------------------
#===================================================================================================================================
#===================================================================================================================================

from source.PathNet2 import Trainer 

from source.california_housing_utils import get_california_housing_data
from source.general_utils import plot_final_loss_distribution, plot_mean_loss_with_std

import torch.nn as nn
import numpy as np


ITERATIONS = 100

RUNS = 2

# MLP Hyperparameters
INPUT_SIZE = 8
HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 32
OUTPUT_SIZE = 1

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
                             quantization_factor=10,
                             parameter_range=(-10, 10),
                             debug_mlp=False,
                             #----------------------------------------------------------------------------------
                             weight_kernel = [4,4], bias_kernel = [4], x_stride=4, y_stride=4, delta_abs=None,
                             #----------------------------------------------------------------------------------
                             early_stopping=True, e_s_patience=250,
                             #----------------------------------------------------------------------------------
                             dynamic_quantization=True, d_q_patience=100, 
                             quantization_factor_multiplier=10, max_quantization_factor=1e4,
                             #-----------------------------------------------------------------------------------
                             dynamic_kernel_reshaping=True, d_k_r_patience=100, 
                             x_weight_kernel_decr=1, y_weight_kernel_decr=1, y_bias_kernel_decr=1, 
                             min_weight_kernel=[2,2], min_bias_kernel=[2],
                             x_stride_decr=1, y_stride_decr=1, min_x_stride=2, min_y_stride=2,
                             #----------------------------------------------------------------------------------
                             loss_improvement_threshold=1e-5,
                             #----------------------------------------------------------------------------------
                             max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=False, model_name=f'housing_dynamic_astar_run_{run + 1}'
                             )

    trainer.train(X_train, Y_train)

    DYNAMIC_ASTAR_METRICS["losses"].append(trainer.loss_history)
    DYNAMIC_ASTAR_METRICS["training_times"].append(trainer.training_time)
    DYNAMIC_ASTAR_METRICS["final_losses"].append(trainer.best_node.h_val)

#    trainer.log_to_txt_file(f"{LOG_FILE_ASTAR}_run_{run + 1}.txt")


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
                             quantization_factor=10,
                             parameter_range=(-10, 10),
                             debug_mlp=False,
                             #----------------------------------------------------------------------------------
                             weight_kernel = [4,4], bias_kernel = [4], x_stride=4, y_stride=4, delta_abs=None,
                             #----------------------------------------------------------------------------------
                             early_stopping=False, e_s_patience=250,
                             #----------------------------------------------------------------------------------
                             dynamic_quantization=False, d_q_patience=100, 
                             quantization_factor_multiplier=10, max_quantization_factor=1e4,
                             #-----------------------------------------------------------------------------------
                             dynamic_kernel_reshaping=False, d_k_r_patience=100, 
                             x_weight_kernel_decr=1, y_weight_kernel_decr=1, y_bias_kernel_decr=1, 
                             min_weight_kernel=[1,1], min_bias_kernel=[1],
                             x_stride_decr=0, y_stride_decr=0, min_x_stride=1, min_y_stride=1,
                             #----------------------------------------------------------------------------------
                             loss_improvement_threshold=1e-5,
                             #----------------------------------------------------------------------------------
                             max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=False, model_name=f'housing_dynamic_astar_run_{run + 1}'
                             )

    trainer.train(X_train, Y_train)

    STATIC_ASTAR_METRICS["losses"].append(trainer.loss_history)
    STATIC_ASTAR_METRICS["training_times"].append(trainer.training_time)
    STATIC_ASTAR_METRICS["final_losses"].append(trainer.best_node.h_val)

#    trainer.log_to_txt_file(f"{LOG_FILE_ASTAR}_run_{run + 1}.txt")


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------


# FINAL LOSS STATS (for Summary Table and Box Plot)
static_astar_final_losses = np.array(STATIC_ASTAR_METRICS["final_losses"])
static_astar_training_times = np.array(STATIC_ASTAR_METRICS["training_times"])
dynamic_astar_final_losses = np.array(DYNAMIC_ASTAR_METRICS["final_losses"])
dynamic_astar_training_times = np.array(DYNAMIC_ASTAR_METRICS["training_times"])


# STATIC A-Star Statistics
static_astar_avg_loss = np.mean(static_astar_final_losses)
static_astar_std_dev = np.std(static_astar_final_losses)
static_astar_variance = np.var(static_astar_final_losses)
static_astar_median = np.median(static_astar_final_losses)
static_astar_min = np.min(static_astar_final_losses)
static_astar_max = np.max(static_astar_final_losses)
static_astar_avg_training_time = np.mean(static_astar_training_times)

# DYNAMIC A-Star Statistics
dynamic_astar_avg_loss = np.mean(dynamic_astar_final_losses)
dynamic_astar_std_dev = np.std(dynamic_astar_final_losses)
dynamic_astar_variance = np.var(dynamic_astar_final_losses)
dynamic_astar_median = np.median(dynamic_astar_final_losses)
dynamic_astar_min = np.min(dynamic_astar_final_losses)
dynamic_astar_max = np.max(dynamic_astar_final_losses)
dynamic_astar_avg_training_time = np.mean(dynamic_astar_training_times)


print("\n=========================================================================================")
print(f"| STATISTICAL SUMMARY over {RUNS} Runs |")
print("=========================================================================================")
print("| Metric      | STATIC A-Star  | DYNAMIC A-Star |")
print("|-------------|----------------|----------------------------|")
print(f"| Average Loss| {static_astar_avg_loss:.6f}     | {dynamic_astar_avg_loss:.6f}              |")
print(f"| Median Loss | {static_astar_median:.6f}     | {dynamic_astar_median:.6f}              |")
print(f"| Std Dev     | {static_astar_std_dev:.6f}     | {dynamic_astar_std_dev:.6f}              |")
print(f"| Variance    | {static_astar_variance:.6f}     | {dynamic_astar_variance:.6f}              |")
print(f"| Min Loss    | {static_astar_min:.6f}     | {dynamic_astar_min:.6f}              |")
print(f"| Max Loss    | {static_astar_max:.6f}     | {dynamic_astar_max:.6f}              |")
print(f"| AVG Training Time | {static_astar_avg_training_time:.6f} | {dynamic_astar_avg_training_time:.6f}         |")
print("=========================================================================================")

with open(f"housing_training_statistics_summary_{RUNS}_runs.txt", "w") as f:
    f.write("=========================================================================================\n")
    f.write(f"| STATISTICAL SUMMARY over {RUNS} Runs |\n")
    f.write("=========================================================================================\n")
    f.write("| Metric      | STATIC A-Star  | DYNAMIC A-Star |\n")
    f.write("|-------------|----------------|----------------------------|\n")
    f.write(f"| Average Loss| {static_astar_avg_loss:.6f}     | {dynamic_astar_avg_loss:.6f}              |\n")
    f.write(f"| Median Loss | {static_astar_median:.6f}     | {dynamic_astar_median:.6f}              |\n")
    f.write(f"| Std Dev     | {static_astar_std_dev:.6f}     | {dynamic_astar_std_dev:.6f}              |\n")
    f.write(f"| Variance    | {static_astar_variance:.6f}     | {dynamic_astar_variance:.6f}              |\n")
    f.write(f"| Min Loss    | {static_astar_min:.6f}     | {dynamic_astar_min:.6f}              |\n")
    f.write(f"| Max Loss    | {static_astar_max:.6f}     | {dynamic_astar_max:.6f}              |\n")
    f.write(f"| AVG Training Time | {static_astar_avg_training_time:.6f} | {dynamic_astar_avg_training_time:.6f}         |\n")
    f.write("=========================================================================================\n")

print(f"\nSaved statistical summary to 'housing_training_statistics_summary_{RUNS}_runs.txt'\n")



static_astar_losses_array = STATIC_ASTAR_METRICS["losses"]
dynamic_astar_losses_array = DYNAMIC_ASTAR_METRICS["losses"]

# mean and standard deviation across all runs for each iteration
static_astar_mean_loss = np.mean(static_astar_losses_array, axis=0)
static_astar_std_loss = np.std(static_astar_losses_array, axis=0)

dynamic_astar_mean_loss = np.mean(dynamic_astar_losses_array, axis=0)
dynamic_astar_std_loss = np.std(dynamic_astar_losses_array, axis=0)

labels = ["STATIC A-Star", "DYNAMIC A-Star"]

# mean loss with standard deviation shading
plot_mean_loss_with_std(labels, static_astar_mean_loss, static_astar_std_loss, dynamic_astar_mean_loss, dynamic_astar_std_loss, RUNS, "housing_mean_loss_comparison_with_std.png", "California Housing")

# box and whisker of final losses
plot_final_loss_distribution(labels, static_astar_final_losses, dynamic_astar_final_losses, RUNS, "housing_final_loss_distribution_comparison.png", "California Housing")