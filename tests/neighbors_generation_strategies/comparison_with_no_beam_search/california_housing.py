#==============================================================================================================================================================
#==============================================================================================================================================================
# python -u -m tests.neighbors_generation_strategies.comparison_with_no_beam_search.california_housing |& tee tests/neighbors_generation_strategies/comparison_with_no_beam_search/california_housing_output.txt 
#==============================================================================================================================================================
#==============================================================================================================================================================

#==============================================================================================================================================================
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                                       Test on California Housing Dataset to compare:
#                     Single Kernel Neighbor Generation vs Layer-Wise Kernels Neighbor Generation vs Random Sampling Neighbor Generation 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#==============================================================================================================================================================


import torch.nn as nn

from source.PathNet import Trainer, TrainerLayerWiseKernel, TrainerRandomSampling
from source.utils.dataset_utils.housing_utils import get_california_housing_data
from source.utils.evaluation_utils import evaluate_pathnet_regression
from source.utils.plot_utils import generate_plots, format_sci,  save_metrics, generate_evaluation_statistical_summary, plot_regression_statistics

ITERATIONS = 2000
RUNS = 5
SAVE_TRAINED_MODEL = False
MODEL_NAME_PREFIX = "housing_model"
DATASET_NAME = "California Housing"
EARLY_STOPPING = False
E_S_PATIENCE = 200
LOSS_IMPROVEMENT_THRESHOLD = 1e-3
PARAMETER_RANGE = (-10, 10)


# =========================================================================================================================================================
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------- SMALL NET --------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================================================================================================================

TEST_NAME = "small_net_neighbors_generation_strategies_comparison_no_BS"

# MLP Hyperparameters
INPUT_SIZE = 8
HIDDEN_SIZE_1 = 32
HIDDEN_SIZE_2 = 16
OUTPUT_SIZE = 1

# Initial Kernel and Stride Settings
WEIGHT_KERNEL = [2,2]
BIAS_KERNEL = [2]
X_STRIDE = 2
Y_STRIDE = 2
DELTA_ABS = None

# Parameters for Layer-Wise Kernels Neighbors Generation
WEIGHT_KERNELS = [[2,2], [2,2], [1,2]]
BIAS_KERNELS = [[2], [2], [1]]
WEIGHT_STRIDES = [[2,2], [2,2], [2,1]]  # Format: list of [x_stride, y_stride] per layer
BIAS_STRIDES = [[2], [2], [1]]            # Format: list of [stride] per layer

# Parameters for Random Sampling Neighbors Generation
PERTURBATION_RATIO = 0.01       # 1% of the parameters will be perturbed per each neighbor
SEARCH_COVERAGE_RATIO = 0.1     # 10% of the total number of parameters in the model will be the number of neighbors generated per each state

# Quantization Parameter
QUANTIZATION_FACTOR = 10

labels_list = ["Single Kernel Neighbors Generation", "layer-Wise Kernels Neighbors Generation", "Random Neighbors Generation"]

metrics_list = [
    {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "dynamic_quantization_iterations": [],
        "dynamic_kernel_reshaping_iterations": [],
        "evaluation_scores": []
    } for _ in range(len(labels_list))
]

X_train, Y_train, X_val, Y_val, X_test, Y_test = get_california_housing_data()
print(f"\nTraining Data Shape: {X_train.shape}, {Y_train.shape}")
print(f"Validation Data Shape: {X_val.shape}, {Y_val.shape}")
print(f"Testing Data Shape: {X_test.shape}, {Y_test.shape}\n")

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------- SINGLE KERNEL NEIGHBORS GENERATION ------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Single Kernel Neighbors Generation \t Vanilla ASTAR Training Run {run + 1} ---\n")

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
                            dynamic_quantization=False,
                            #-----------------------------------------------------------------------------------
                            dynamic_kernel_reshaping=False,
                            #----------------------------------------------------------------------------------
                            loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                            #----------------------------------------------------------------------------------
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_single_kernel_astar_run_{run + 1}'
                            )

    trainer.train(X_train, Y_train)

    metrics_list[0]["losses"].append(trainer.loss_history)
    metrics_list[0]["training_times"].append(trainer.training_time)
    metrics_list[0]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[0]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    metrics_list[0]["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])
    metrics_list[0]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------- LAYER-WISE KERNELS NEIGHBORS GENERATION ----------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Layer-Wise Kernels Neighbors Generation \t Vanilla ASTAR Training Run {run + 1} ---\n")
    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, OUTPUT_SIZE),
            )


    trainer = TrainerLayerWiseKernel(model=model,
                            loss_fn=nn.MSELoss(),
                            quantization_factor=QUANTIZATION_FACTOR,
                            parameter_range=PARAMETER_RANGE,
                            debug_mlp=False,
                            #----------------------------------------------------------------------------------
                            weight_kernels = WEIGHT_KERNELS, bias_kernels = BIAS_KERNELS, weight_strides=WEIGHT_STRIDES, bias_strides=BIAS_STRIDES, delta_abs=DELTA_ABS,
                            #----------------------------------------------------------------------------------
                            early_stopping=EARLY_STOPPING, e_s_patience=E_S_PATIENCE,
                            #----------------------------------------------------------------------------------
                            dynamic_quantization=False,
                            #-----------------------------------------------------------------------------------
                            dynamic_kernel_reshaping=False,
                            #----------------------------------------------------------------------------------
                            loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                            #----------------------------------------------------------------------------------
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_layer_wise_kernels_astar_run_{run + 1}'
                            )

    trainer.train(X_train, Y_train)

    metrics_list[1]["losses"].append(trainer.loss_history)
    metrics_list[1]["training_times"].append(trainer.training_time)
    metrics_list[1]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[1]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    metrics_list[1]["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])
    metrics_list[1]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------- RANDOM SAMPLING NEIGHBORS GENERATION -------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Random Sampling Neighbors Generation \t Vanilla ASTAR Training Run {run + 1} ---\n")
    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, OUTPUT_SIZE),
            )


    trainer = TrainerRandomSampling(model=model,
                            loss_fn=nn.MSELoss(),
                            quantization_factor=QUANTIZATION_FACTOR,
                            parameter_range=PARAMETER_RANGE,
                            debug_mlp=False,
                            #----------------------------------------------------------------------------------
                            perturbation_ratio=PERTURBATION_RATIO, search_coverage_ratio=SEARCH_COVERAGE_RATIO,
                            delta_abs=DELTA_ABS,
                            #----------------------------------------------------------------------------------
                            early_stopping=EARLY_STOPPING, e_s_patience=E_S_PATIENCE,
                            #----------------------------------------------------------------------------------
                            dynamic_quantization=False,
                            #----------------------------------------------------------------------------------
                            loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                            #----------------------------------------------------------------------------------
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_random_sampling_astar_run_{run + 1}'
                            )

    trainer.train(X_train, Y_train)

    metrics_list[2]["losses"].append(trainer.loss_history)
    metrics_list[2]["training_times"].append(trainer.training_time)
    metrics_list[2]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[2]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    metrics_list[2]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


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

TEST_NAME = "medium_net_neighbors_generation_strategies_comparison_no_BS"

# MLP Hyperparameters
HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 32
HIDDEN_SIZE_3 = 16

# Initial Kernel and Stride Settings
WEIGHT_KERNEL = [4,4]
BIAS_KERNEL = [4]
X_STRIDE = 4
Y_STRIDE = 4

# Parameters for Layer-Wise Kernels Neighbors Generation
WEIGHT_KERNELS = [[4,4], [4,4], [4,4], [1,4]]
BIAS_KERNELS = [[4], [4], [4], [1]]
WEIGHT_STRIDES = [[4,4], [4,4], [4,4], [4,1]]      # Format: list of [x_stride, y_stride] per layer
BIAS_STRIDES = [[4], [4], [4], [1]]                # Format: list of [stride] per layer

# Parameters for Random Sampling Neighbors Generation
PERTURBATION_RATIO = 0.01        # 1% of the parameters will be perturbed per each neighbor
SEARCH_COVERAGE_RATIO = 0.05     # 5% of the total number of parameters in the model will be the number of neighbors generated per each state

labels_list = ["Single Kernel Neighbors Generation", "layer-Wise Kernels Neighbors Generation", "Random Neighbors Generation"]

metrics_list = [
    {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "dynamic_quantization_iterations": [],
        "dynamic_kernel_reshaping_iterations": [],
        "evaluation_scores": []
    } for _ in range(len(labels_list))
]

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------- SINGLE KERNEL NEIGHBORS GENERATION ------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Single Kernel Neighbors Generation \t Vanilla ASTAR Training Run {run + 1} ---\n")

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
                            dynamic_quantization=False,
                            #-----------------------------------------------------------------------------------
                            dynamic_kernel_reshaping=False,
                            #----------------------------------------------------------------------------------
                            loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                            #----------------------------------------------------------------------------------
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_single_kernel_astar_run_{run + 1}'
                            )

    trainer.train(X_train, Y_train)

    metrics_list[0]["losses"].append(trainer.loss_history)
    metrics_list[0]["training_times"].append(trainer.training_time)
    metrics_list[0]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[0]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    metrics_list[0]["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])
    metrics_list[0]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------- LAYER-WISE KERNELS NEIGHBORS GENERATION ----------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Layer-Wise Kernels Neighbors Generation \t Vanilla ASTAR Training Run {run + 1} ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_3),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_3, OUTPUT_SIZE),
            )


    trainer = TrainerLayerWiseKernel(model=model,
                            loss_fn=nn.MSELoss(),
                            quantization_factor=QUANTIZATION_FACTOR,
                            parameter_range=PARAMETER_RANGE,
                            debug_mlp=False,
                            #----------------------------------------------------------------------------------
                            weight_kernels = WEIGHT_KERNELS, bias_kernels = BIAS_KERNELS, weight_strides=WEIGHT_STRIDES, bias_strides=BIAS_STRIDES, delta_abs=DELTA_ABS,
                            #----------------------------------------------------------------------------------
                            early_stopping=EARLY_STOPPING, e_s_patience=E_S_PATIENCE,
                            #----------------------------------------------------------------------------------
                            dynamic_quantization=False,
                            #-----------------------------------------------------------------------------------
                            dynamic_kernel_reshaping=False,
                            #----------------------------------------------------------------------------------
                            loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                            #----------------------------------------------------------------------------------
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_layer_wise_kernels_astar_run_{run + 1}'
                            )

    trainer.train(X_train, Y_train)

    metrics_list[1]["losses"].append(trainer.loss_history)
    metrics_list[1]["training_times"].append(trainer.training_time)
    metrics_list[1]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[1]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    metrics_list[1]["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])
    metrics_list[1]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------- RANDOM SAMPLING NEIGHBORS GENERATION -------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Random Sampling Neighbors Generation \t Vanilla ASTAR Training Run {run + 1} ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_3),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_3, OUTPUT_SIZE),
            )


    trainer = TrainerRandomSampling(model=model,
                            loss_fn=nn.MSELoss(),
                            quantization_factor=QUANTIZATION_FACTOR,
                            parameter_range=PARAMETER_RANGE,
                            debug_mlp=False,
                            #----------------------------------------------------------------------------------
                            perturbation_ratio=PERTURBATION_RATIO, search_coverage_ratio=SEARCH_COVERAGE_RATIO,
                            delta_abs=DELTA_ABS,
                            #----------------------------------------------------------------------------------
                            early_stopping=EARLY_STOPPING, e_s_patience=E_S_PATIENCE,
                            #----------------------------------------------------------------------------------
                            dynamic_quantization=False,
                            #----------------------------------------------------------------------------------
                            loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                            #----------------------------------------------------------------------------------
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_random_sampling_astar_run_{run + 1}'
                            )

    trainer.train(X_train, Y_train)

    metrics_list[2]["losses"].append(trainer.loss_history)
    metrics_list[2]["training_times"].append(trainer.training_time)
    metrics_list[2]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[2]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    metrics_list[2]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))

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

TEST_NAME = "big_net_neighbors_generation_strategies_comparison_no_BS"

# MLP Hyperparameters
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64
HIDDEN_SIZE_3 = 32
HIDDEN_SIZE_4 = 16

# Initial Kernel and Stride Settings
WEIGHT_KERNEL = [6,6]
BIAS_KERNEL = [6]
X_STRIDE = 6
Y_STRIDE = 6

# Parameters for Layer-Wise Kernels Neighbors Generation
WEIGHT_KERNELS = [[6,2], [6,6], [4,4], [4,4], [1,2]]
BIAS_KERNELS = [[6], [6], [4], [2], [1]]
WEIGHT_STRIDES = [[2,6], [6,6], [4,4], [4,4], [2,1]]      # Format: list of [x_stride, y_stride] per layer
BIAS_STRIDES = [[6], [6], [4], [2], [1]]                  # Format: list of [stride] per layer

# Parameters for Random Sampling Neighbors Generation
PERTURBATION_RATIO = 0.1         # 10% of the parameters will be perturbed per each neighbor
SEARCH_COVERAGE_RATIO = 0.05     # 5% of the total number of parameters in the model will be the number of neighbors generated per each state

labels_list = ["Single Kernel Neighbors Generation", "layer-Wise Kernels Neighbors Generation", "Random Neighbors Generation"]

metrics_list = [
    {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "dynamic_quantization_iterations": [],
        "dynamic_kernel_reshaping_iterations": [],
        "evaluation_scores": []
    } for _ in range(len(labels_list))
]

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------- SINGLE KERNEL NEIGHBORS GENERATION ------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Single Kernel Neighbors Generation \t Vanilla ASTAR Training Run {run + 1} ---\n")

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
                            dynamic_quantization=False,
                            #-----------------------------------------------------------------------------------
                            dynamic_kernel_reshaping=False,
                            #----------------------------------------------------------------------------------
                            loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                            #----------------------------------------------------------------------------------
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_vanilla_astar_run_{run + 1}'
                            )

    trainer.train(X_train, Y_train)

    metrics_list[0]["losses"].append(trainer.loss_history)
    metrics_list[0]["training_times"].append(trainer.training_time)
    metrics_list[0]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[0]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    metrics_list[0]["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])
    metrics_list[0]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------- LAYER-WISE KERNELS NEIGHBORS GENERATION ----------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Layer-Wise Kernels Neighbors Generation \t Vanilla ASTAR Training Run {run + 1} ---\n")

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


    trainer = TrainerLayerWiseKernel(model=model,
                            loss_fn=nn.MSELoss(),
                            quantization_factor=QUANTIZATION_FACTOR,
                            parameter_range=PARAMETER_RANGE,
                            debug_mlp=False,
                            #----------------------------------------------------------------------------------
                            weight_kernels = WEIGHT_KERNELS, bias_kernels = BIAS_KERNELS, weight_strides=WEIGHT_STRIDES, bias_strides=BIAS_STRIDES, delta_abs=DELTA_ABS,
                            #----------------------------------------------------------------------------------
                            early_stopping=EARLY_STOPPING, e_s_patience=E_S_PATIENCE,
                            #----------------------------------------------------------------------------------
                            dynamic_quantization=False,
                            #-----------------------------------------------------------------------------------
                            dynamic_kernel_reshaping=False,
                            #----------------------------------------------------------------------------------
                            loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                            #----------------------------------------------------------------------------------
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_layer_wise_kernels_astar_run_{run + 1}'
                            )

    trainer.train(X_train, Y_train)

    metrics_list[1]["losses"].append(trainer.loss_history)
    metrics_list[1]["training_times"].append(trainer.training_time)
    metrics_list[1]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[1]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    metrics_list[1]["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])
    metrics_list[1]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------- RANDOM SAMPLING NEIGHBORS GENERATION -------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Random Sampling Neighbors Generation \t Vanilla ASTAR Training Run {run + 1} ---\n")

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


    trainer = TrainerRandomSampling(model=model,
                            loss_fn=nn.MSELoss(),
                            quantization_factor=QUANTIZATION_FACTOR,
                            parameter_range=PARAMETER_RANGE,
                            debug_mlp=False,
                            #----------------------------------------------------------------------------------
                            perturbation_ratio=PERTURBATION_RATIO, search_coverage_ratio=SEARCH_COVERAGE_RATIO,
                            delta_abs=DELTA_ABS,
                            #----------------------------------------------------------------------------------
                            early_stopping=EARLY_STOPPING, e_s_patience=E_S_PATIENCE,
                            #----------------------------------------------------------------------------------
                            dynamic_quantization=False,
                            #----------------------------------------------------------------------------------
                            loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                            #----------------------------------------------------------------------------------
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_random_sampling_astar_run_{run + 1}'
                            )

    trainer.train(X_train, Y_train)

    metrics_list[2]["losses"].append(trainer.loss_history)
    metrics_list[2]["training_times"].append(trainer.training_time)
    metrics_list[2]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[2]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    metrics_list[2]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

all_results = {label: metric for label, metric in zip(labels_list, metrics_list)}

save_metrics(all_results, TEST_NAME)

generate_evaluation_statistical_summary(metrics_list,labels_list, TEST_NAME)

generate_plots(metrics_list, labels_list, TEST_NAME, DATASET_NAME)

plot_regression_statistics(metrics_list, labels_list, TEST_NAME, DATASET_NAME)
