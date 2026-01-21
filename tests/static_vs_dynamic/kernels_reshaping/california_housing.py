#==============================================================================================================================================================
#==============================================================================================================================================================
# python -u -m tests.static_vs_dynamic.kernels_reshaping.california_housing |& tee tests/static_vs_dynamic/kernels_reshaping/california_housing_output.txt 
#==============================================================================================================================================================
#==============================================================================================================================================================

#==============================================================================================================================================================
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                   Test on California Housing Dataset to compare Dynamic vs Static Kernel Reshaping A-star Training
#                   Each MLP architecture is trained multiple times with varying kernel sizes for the static approach, 
#                       while the dynamic approach starts with a maximum kernel size and reduces it over time.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#==============================================================================================================================================================


import torch.nn as nn

from source.PathNet import Trainer
from source.utils.dataset_utils.housing_utils import get_california_housing_data
from source.utils.evaluation_utils import evaluate_pathnet_regression
from source.utils.plot_utils import generate_plots,  save_metrics, generate_evaluation_statistical_summary, plot_regression_statistics

ITERATIONS = 2000
RUNS = 5
SAVE_TRAINED_MODEL = False
MODEL_NAME_PREFIX = "housing_model"
DATASET_NAME = "California Housing"
EARLY_STOPPING = False
E_S_PATIENCE = 200
LOSS_IMPROVEMENT_THRESHOLD = 1e-3
PARAMETER_RANGE = (-10, 10)
QUANTIZATION_FACTOR = 10


# =========================================================================================================================================================
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------- SMALL NET --------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================================================================================================================

TEST_NAME = "small_net_dynamic_kernels_reshaping"

# MLP Hyperparameters
INPUT_SIZE = 8
HIDDEN_SIZE_1 = 32
HIDDEN_SIZE_2 = 16
OUTPUT_SIZE = 1

# Initial Kernel and Stride Settings
MAX_WEIGHT_KERNEL = [4,4]
MAX_BIAS_KERNEL = [4]
MAX_X_STRIDE = 4
MAX_Y_STRIDE = 4
DELTA_ABS = None

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

metrics_list = [
    {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "dynamic_quantization_iterations": [],
        "dynamic_kernel_reshaping_iterations": [],
        "evaluation_scores": []
    }
]

for kernel_size in range(MAX_WEIGHT_KERNEL[0], MIN_WEIGHT_KERNEL[0] - 1, -1):
    STATIC_ASTAR_METRICS = {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "evaluation_scores": []
    }
    metrics_list.append(STATIC_ASTAR_METRICS)

labels_list = [f"Dyn. KS=[{MAX_WEIGHT_KERNEL[0]}x{MAX_WEIGHT_KERNEL[1]}-{MIN_WEIGHT_KERNEL[0]}x{MIN_WEIGHT_KERNEL[1]}]"]
for kernel_size in range(MAX_WEIGHT_KERNEL[0], MIN_WEIGHT_KERNEL[0] - 1, -1):
    labels_list.append(f"Stat. KS={kernel_size}x{kernel_size}")
                       
X_train, Y_train, X_val, Y_val, X_test, Y_test = get_california_housing_data()
print(f"\nTraining Data Shape: {X_train.shape}, {Y_train.shape}")
print(f"Validation Data Shape: {X_val.shape}, {Y_val.shape}")
print(f"Testing Data Shape: {X_test.shape}, {Y_test.shape}\n")

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- DYNAMIC ASTAR TRAINING ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t DYNAMIC ASTAR Training Run {run + 1} \t Kernel Size: [{MAX_WEIGHT_KERNEL[0]}x{MAX_WEIGHT_KERNEL[1]} - {MIN_WEIGHT_KERNEL[0]}x{MIN_WEIGHT_KERNEL[1]} ] ---\n")

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
                             weight_kernel = MAX_WEIGHT_KERNEL, bias_kernel = MAX_BIAS_KERNEL, x_stride=MAX_X_STRIDE, y_stride=MAX_Y_STRIDE, delta_abs=DELTA_ABS,
                             #----------------------------------------------------------------------------------
                             early_stopping=EARLY_STOPPING, e_s_patience=E_S_PATIENCE,
                             #----------------------------------------------------------------------------------
                             dynamic_quantization=False,
                             #-----------------------------------------------------------------------------------
                             dynamic_kernel_reshaping=True, d_k_r_patience=D_K_R_PATIENCE, 
                             x_weight_kernel_decr=X_WEIGHT_KERNEL_DECR, y_weight_kernel_decr=Y_WEIGHT_KERNEL_DECR, y_bias_kernel_decr=Y_BIAS_KERNEL_DECR, 
                             min_weight_kernel=MIN_WEIGHT_KERNEL, min_bias_kernel=MIN_BIAS_KERNEL,
                             x_stride_decr=X_STRIDE_DECR, y_stride_decr=Y_STRIDE_DECR, min_x_stride=MIN_X_STRIDE, min_y_stride=MIN_Y_STRIDE,
                             #----------------------------------------------------------------------------------
                             loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                             #----------------------------------------------------------------------------------
                             max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_dynamic_astar_run_{run + 1}'
                             )

    trainer.train(X_train, Y_train)

    metrics_list[0]["losses"].append(trainer.loss_history)
    metrics_list[0]["training_times"].append(trainer.training_time)
    metrics_list[0]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[0]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    metrics_list[0]["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])
    metrics_list[0]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- STATIC ASTAR TRAININGS ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for index, kernel_size in enumerate(range(MAX_WEIGHT_KERNEL[0], MIN_WEIGHT_KERNEL[0] - 1, -1)):
    for run in range(RUNS):
        print(f"\n--- TEST NAME: {TEST_NAME} \t STATIC ASTAR Training Run {run + 1} \t Kernel Size: [{int(kernel_size)} x {int(kernel_size)}] ---\n")

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
                            weight_kernel = [kernel_size, kernel_size], bias_kernel = [kernel_size], x_stride=kernel_size, y_stride=kernel_size, delta_abs=DELTA_ABS,
                            #----------------------------------------------------------------------------------
                            early_stopping=EARLY_STOPPING, e_s_patience=E_S_PATIENCE,
                            #----------------------------------------------------------------------------------
                            dynamic_quantization=False,
                            #-----------------------------------------------------------------------------------
                            dynamic_kernel_reshaping=False,
                            #----------------------------------------------------------------------------------
                            loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                            #----------------------------------------------------------------------------------
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_static_astar_ks_{kernel_size}_run_{run + 1}'
                            )

        trainer.train(X_train, Y_train)

        metrics_list[index + 1]["losses"].append(trainer.loss_history)
        metrics_list[index + 1]["training_times"].append(trainer.training_time)
        metrics_list[index + 1]["final_losses"].append(trainer.best_node.h_val)
        metrics_list[index + 1]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------


all_results = {label: metric for label, metric in zip(labels_list, metrics_list)}

save_metrics(all_results, TEST_NAME)

generate_evaluation_statistical_summary(metrics_list,labels_list, TEST_NAME)

generate_plots(metrics_list, labels_list, TEST_NAME, DATASET_NAME)

plot_regression_statistics(metrics_list, labels_list, TEST_NAME, DATASET_NAME)




# =========================================================================================================================================================
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------- MEDIUM NET --------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================================================================================================================

TEST_NAME = "medium_net_dynamic_kernels_reshaping"

# MLP Hyperparameters
HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 32
HIDDEN_SIZE_3 = 16

# Initial Kernel and Stride Settings
MAX_WEIGHT_KERNEL = [6,6]
MAX_BIAS_KERNEL = [6]
MAX_X_STRIDE = 6
MAX_Y_STRIDE = 6

# Dynamic Kernel Reshaping Settings
MIN_WEIGHT_KERNEL = [2,2]
MIN_BIAS_KERNEL = [2]
MIN_X_STRIDE = 2
MIN_Y_STRIDE = 2

metrics_list = [
    {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "dynamic_quantization_iterations": [],
        "dynamic_kernel_reshaping_iterations": [],
        "evaluation_scores": []
    }
]

for kernel_size in range(MAX_WEIGHT_KERNEL[0], MIN_WEIGHT_KERNEL[0] - 1, -1):
    STATIC_ASTAR_METRICS = {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "evaluation_scores": []
    }
    metrics_list.append(STATIC_ASTAR_METRICS)

labels_list = [f"Dyn. KS=[{MAX_WEIGHT_KERNEL[0]}x{MAX_WEIGHT_KERNEL[1]}-{MIN_WEIGHT_KERNEL[0]}x{MIN_WEIGHT_KERNEL[1]}]"]
for kernel_size in range(MAX_WEIGHT_KERNEL[0], MIN_WEIGHT_KERNEL[0] - 1, -1):
    labels_list.append(f"Stat. KS={kernel_size}x{kernel_size}")
                       
X_train, Y_train, X_val, Y_val, X_test, Y_test = get_california_housing_data()
print(f"\nTraining Data Shape: {X_train.shape}, {Y_train.shape}")
print(f"Validation Data Shape: {X_val.shape}, {Y_val.shape}")
print(f"Testing Data Shape: {X_test.shape}, {Y_test.shape}\n")

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- DYNAMIC ASTAR TRAINING ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t DYNAMIC ASTAR Training Run {run + 1} \t Kernel Size: [{MAX_WEIGHT_KERNEL[0]}x{MAX_WEIGHT_KERNEL[1]} - {MIN_WEIGHT_KERNEL[0]}x{MIN_WEIGHT_KERNEL[1]} ] ---\n")

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
                             weight_kernel = MAX_WEIGHT_KERNEL, bias_kernel = MAX_BIAS_KERNEL, x_stride=MAX_X_STRIDE, y_stride=MAX_Y_STRIDE, delta_abs=DELTA_ABS,
                             #----------------------------------------------------------------------------------
                             early_stopping=EARLY_STOPPING, e_s_patience=E_S_PATIENCE,
                             #----------------------------------------------------------------------------------
                             dynamic_quantization=False,
                             #-----------------------------------------------------------------------------------
                             dynamic_kernel_reshaping=True, d_k_r_patience=D_K_R_PATIENCE, 
                             x_weight_kernel_decr=X_WEIGHT_KERNEL_DECR, y_weight_kernel_decr=Y_WEIGHT_KERNEL_DECR, y_bias_kernel_decr=Y_BIAS_KERNEL_DECR, 
                             min_weight_kernel=MIN_WEIGHT_KERNEL, min_bias_kernel=MIN_BIAS_KERNEL,
                             x_stride_decr=X_STRIDE_DECR, y_stride_decr=Y_STRIDE_DECR, min_x_stride=MIN_X_STRIDE, min_y_stride=MIN_Y_STRIDE,
                             #----------------------------------------------------------------------------------
                             loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                             #----------------------------------------------------------------------------------
                             max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_dynamic_astar_run_{run + 1}'
                             )

    trainer.train(X_train, Y_train)

    metrics_list[0]["losses"].append(trainer.loss_history)
    metrics_list[0]["training_times"].append(trainer.training_time)
    metrics_list[0]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[0]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    metrics_list[0]["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])
    metrics_list[0]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- STATIC ASTAR TRAININGS ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for index, kernel_size in enumerate(range(MAX_WEIGHT_KERNEL[0], MIN_WEIGHT_KERNEL[0] - 1, -1)):
    for run in range(RUNS):
        print(f"\n--- TEST NAME: {TEST_NAME} \t STATIC ASTAR Training Run {run + 1} \t Kernel Size: [{int(kernel_size)} x {int(kernel_size)}] ---\n")

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
                            weight_kernel = [kernel_size, kernel_size], bias_kernel = [kernel_size], x_stride=kernel_size, y_stride=kernel_size, delta_abs=DELTA_ABS,
                            #----------------------------------------------------------------------------------
                            early_stopping=EARLY_STOPPING, e_s_patience=E_S_PATIENCE,
                            #----------------------------------------------------------------------------------
                            dynamic_quantization=False,
                            #-----------------------------------------------------------------------------------
                            dynamic_kernel_reshaping=False,
                            #----------------------------------------------------------------------------------
                            loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                            #----------------------------------------------------------------------------------
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_static_astar_ks_{kernel_size}_run_{run + 1}'
                            )

        trainer.train(X_train, Y_train)

        metrics_list[index + 1]["losses"].append(trainer.loss_history)
        metrics_list[index + 1]["training_times"].append(trainer.training_time)
        metrics_list[index + 1]["final_losses"].append(trainer.best_node.h_val)
        metrics_list[index + 1]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------


all_results = {label: metric for label, metric in zip(labels_list, metrics_list)}

save_metrics(all_results, TEST_NAME)

generate_evaluation_statistical_summary(metrics_list,labels_list, TEST_NAME)

generate_plots(metrics_list, labels_list, TEST_NAME, DATASET_NAME)

plot_regression_statistics(metrics_list, labels_list, TEST_NAME, DATASET_NAME)



# =========================================================================================================================================================
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------- BIG NET ----------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================================================================================================================

TEST_NAME = "big_net_dynamic_kernels_reshaping"

# MLP Hyperparameters
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64
HIDDEN_SIZE_3 = 32
HIDDEN_SIZE_4 = 16

# Initial Kernel and Stride Settings
MAX_WEIGHT_KERNEL = [8,8]
MAX_BIAS_KERNEL = [8]
MAX_X_STRIDE = 8
MAX_Y_STRIDE = 8

# Dynamic Kernel Reshaping Settings
MIN_WEIGHT_KERNEL = [3,3]
MIN_BIAS_KERNEL = [3]
MIN_X_STRIDE = 3
MIN_Y_STRIDE = 3

metrics_list = [
    {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "dynamic_quantization_iterations": [],
        "dynamic_kernel_reshaping_iterations": [],
        "evaluation_scores": []
    }
]

for kernel_size in range(MAX_WEIGHT_KERNEL[0], MIN_WEIGHT_KERNEL[0] - 1, -1):
    STATIC_ASTAR_METRICS = {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "evaluation_scores": []
    }
    metrics_list.append(STATIC_ASTAR_METRICS)

labels_list = [f"Dyn. KS=[{MAX_WEIGHT_KERNEL[0]}x{MAX_WEIGHT_KERNEL[1]}-{MIN_WEIGHT_KERNEL[0]}x{MIN_WEIGHT_KERNEL[1]}]"]
for kernel_size in range(MAX_WEIGHT_KERNEL[0], MIN_WEIGHT_KERNEL[0] - 1, -1):
    labels_list.append(f"Stat. KS={kernel_size}x{kernel_size}")
                       
X_train, Y_train, X_val, Y_val, X_test, Y_test = get_california_housing_data()
print(f"\nTraining Data Shape: {X_train.shape}, {Y_train.shape}")
print(f"Validation Data Shape: {X_val.shape}, {Y_val.shape}")
print(f"Testing Data Shape: {X_test.shape}, {Y_test.shape}\n")

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- DYNAMIC ASTAR TRAINING ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t DYNAMIC ASTAR Training Run {run + 1} \t Kernel Size: [{MAX_WEIGHT_KERNEL[0]}x{MAX_WEIGHT_KERNEL[1]} - {MIN_WEIGHT_KERNEL[0]}x{MIN_WEIGHT_KERNEL[1]} ] ---\n")

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
                             weight_kernel = MAX_WEIGHT_KERNEL, bias_kernel = MAX_BIAS_KERNEL, x_stride=MAX_X_STRIDE, y_stride=MAX_Y_STRIDE, delta_abs=DELTA_ABS,
                             #----------------------------------------------------------------------------------
                             early_stopping=EARLY_STOPPING, e_s_patience=E_S_PATIENCE,
                             #----------------------------------------------------------------------------------
                             dynamic_quantization=False,
                             #-----------------------------------------------------------------------------------
                             dynamic_kernel_reshaping=True, d_k_r_patience=D_K_R_PATIENCE, 
                             x_weight_kernel_decr=X_WEIGHT_KERNEL_DECR, y_weight_kernel_decr=Y_WEIGHT_KERNEL_DECR, y_bias_kernel_decr=Y_BIAS_KERNEL_DECR, 
                             min_weight_kernel=MIN_WEIGHT_KERNEL, min_bias_kernel=MIN_BIAS_KERNEL,
                             x_stride_decr=X_STRIDE_DECR, y_stride_decr=Y_STRIDE_DECR, min_x_stride=MIN_X_STRIDE, min_y_stride=MIN_Y_STRIDE,
                             #----------------------------------------------------------------------------------
                             loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                             #----------------------------------------------------------------------------------
                             max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_dynamic_astar_run_{run + 1}'
                             )

    trainer.train(X_train, Y_train)

    metrics_list[0]["losses"].append(trainer.loss_history)
    metrics_list[0]["training_times"].append(trainer.training_time)
    metrics_list[0]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[0]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    metrics_list[0]["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])
    metrics_list[0]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- STATIC ASTAR TRAININGS ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for index, kernel_size in enumerate(range(MAX_WEIGHT_KERNEL[0], MIN_WEIGHT_KERNEL[0] - 1, -1)):
    for run in range(RUNS):
        print(f"\n--- TEST NAME: {TEST_NAME} \t STATIC ASTAR Training Run {run + 1} \t Kernel Size: [{int(kernel_size)} x {int(kernel_size)}] ---\n")

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
                            weight_kernel = [kernel_size, kernel_size], bias_kernel = [kernel_size], x_stride=kernel_size, y_stride=kernel_size, delta_abs=DELTA_ABS,
                            #----------------------------------------------------------------------------------
                            early_stopping=EARLY_STOPPING, e_s_patience=E_S_PATIENCE,
                            #----------------------------------------------------------------------------------
                            dynamic_quantization=False,
                            #-----------------------------------------------------------------------------------
                            dynamic_kernel_reshaping=False,
                            #----------------------------------------------------------------------------------
                            loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                            #----------------------------------------------------------------------------------
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_static_astar_ks_{kernel_size}_run_{run + 1}'
                            )

        trainer.train(X_train, Y_train)

        metrics_list[index + 1]["losses"].append(trainer.loss_history)
        metrics_list[index + 1]["training_times"].append(trainer.training_time)
        metrics_list[index + 1]["final_losses"].append(trainer.best_node.h_val)
        metrics_list[index + 1]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------


all_results = {label: metric for label, metric in zip(labels_list, metrics_list)}

save_metrics(all_results, TEST_NAME)

generate_evaluation_statistical_summary(metrics_list,labels_list, TEST_NAME)

generate_plots(metrics_list, labels_list, TEST_NAME, DATASET_NAME)

plot_regression_statistics(metrics_list, labels_list, TEST_NAME, DATASET_NAME)


