#==============================================================================================================================================================
#==============================================================================================================================================================
# python -u -m tests.neighbors_generation_strategies.grid_search_random_sampling.california_housing |& tee tests/neighbors_generation_strategies/grid_search_random_sampling/california_housing_output.txt 
#==============================================================================================================================================================
#==============================================================================================================================================================

#==============================================================================================================================================================
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               Test on California Housing Dataset to search a good set of parameters for Random Sampling Neighbors Generation
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#==============================================================================================================================================================


import torch.nn as nn

from source.PathNet import TrainerRandomSampling
from source.utils.dataset_utils.housing_utils import get_california_housing_data
from source.utils.evaluation_utils import evaluate_pathnet_regression
from source.utils.plot_utils import generate_plots, format_sci,  save_metrics, generate_evaluation_statistical_summary, plot_regression_statistics

ITERATIONS = 2
RUNS = 1
SAVE_TRAINED_MODEL = False
MODEL_NAME_PREFIX = "housing_model"
DATASET_NAME = "California Housing"
EARLY_STOPPING = False
E_S_PATIENCE = 200
LOSS_IMPROVEMENT_THRESHOLD = 1e-3
PARAMETER_RANGE = (-10, 10)
DELTA_ABS = None

# Parameters for Random Sampling Neighbors Generation
PERTURBATION_RATIOS = [0.01, 0.05, 0.1]       # [1%, 5%, 10%]: percentage of the random sampled parameters (per each layer) that will be perturbed per each neighbor
SEARCH_COVERAGE_RATIOS = [0.05, 0.1, 0.2]     # [5%, 10%, 20%]: percentage of per layer params that determines the total number of neighbors generated per layer.


# =========================================================================================================================================================
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------- SMALL NET --------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================================================================================================================

TEST_NAME = "small_net_random_sampling_grid_search"

# MLP Hyperparameters
INPUT_SIZE = 8
HIDDEN_SIZE_1 = 32
HIDDEN_SIZE_2 = 16
OUTPUT_SIZE = 1

# Quantization Parameter
QUANTIZATION_FACTOR = 10

labels_list = [f'Pert. Ratio: {pr}, Search Ratio: {scr}' for pr in PERTURBATION_RATIOS for scr in SEARCH_COVERAGE_RATIOS]

metrics_list = [
    {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "dynamic_quantization_iterations": [],
        "dynamic_kernel_reshaping_iterations": [],
        "evaluation_scores": []
    } for _ in range(len(PERTURBATION_RATIOS) * len(SEARCH_COVERAGE_RATIOS))
]

X_train, Y_train, X_val, Y_val, X_test, Y_test = get_california_housing_data()
print(f"\nTraining Data Shape: {X_train.shape}, {Y_train.shape}")
print(f"Validation Data Shape: {X_val.shape}, {Y_val.shape}")
print(f"Testing Data Shape: {X_test.shape}, {Y_test.shape}\n")


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------- RANDOM SAMPLING NEIGHBORS GENERATION -------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for p_idx, perturbation_ratio in enumerate(PERTURBATION_RATIOS):
    for s_idx, search_coverage_ratio in enumerate(SEARCH_COVERAGE_RATIOS):
        idx = p_idx * len(SEARCH_COVERAGE_RATIOS) + s_idx
        for run in range(RUNS):
            print(f"\n--- TEST NAME: {TEST_NAME} \t Random Sampling Neighbors Generation \t Perturbation Ratio: {perturbation_ratio}, Search Coverage Ratio: {search_coverage_ratio} \t Vanilla ASTAR Training Run {run + 1} ---\n")
            
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
                                    perturbation_ratio=perturbation_ratio, search_coverage_ratio=search_coverage_ratio,
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

            metrics_list[idx]["losses"].append(trainer.loss_history)
            metrics_list[idx]["training_times"].append(trainer.training_time)
            metrics_list[idx]["final_losses"].append(trainer.best_node.h_val)
            metrics_list[idx]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
            metrics_list[idx]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


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

TEST_NAME = "medium_net_random_sampling_grid_search"

# MLP Hyperparameters
HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 32
HIDDEN_SIZE_3 = 16


labels_list = [f'Pert. Ratio: {pr}, Search Ratio: {scr}' for pr in PERTURBATION_RATIOS for scr in SEARCH_COVERAGE_RATIOS]

metrics_list = [
    {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "dynamic_quantization_iterations": [],
        "dynamic_kernel_reshaping_iterations": [],
        "evaluation_scores": []
    } for _ in range(len(PERTURBATION_RATIOS) * len(SEARCH_COVERAGE_RATIOS))
]

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------- RANDOM SAMPLING NEIGHBORS GENERATION -------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for p_idx, perturbation_ratio in enumerate(PERTURBATION_RATIOS):
    for s_idx, search_coverage_ratio in enumerate(SEARCH_COVERAGE_RATIOS):
        idx = p_idx * len(SEARCH_COVERAGE_RATIOS) + s_idx
        for run in range(RUNS):
            print(f"\n--- TEST NAME: {TEST_NAME} \t Random Sampling Neighbors Generation \t Perturbation Ratio: {perturbation_ratio}, Search Coverage Ratio: {search_coverage_ratio} \t Vanilla ASTAR Training Run {run + 1} ---\n")
            
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
                                    perturbation_ratio=perturbation_ratio, search_coverage_ratio=search_coverage_ratio,
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

            metrics_list[idx]["losses"].append(trainer.loss_history)
            metrics_list[idx]["training_times"].append(trainer.training_time)
            metrics_list[idx]["final_losses"].append(trainer.best_node.h_val)
            metrics_list[idx]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
            metrics_list[idx]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))

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

TEST_NAME = "big_net_random_sampling_grid_search"
# MLP Hyperparameters
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64
HIDDEN_SIZE_3 = 32
HIDDEN_SIZE_4 = 16

labels_list = ["Single Kernel Neighbors Generation", "layer-Wise Kernels Neighbors Generation", "Random Neighbors Generation"]

labels_list = [f'Pert. Ratio: {pr}, Search Ratio: {scr}' for pr in PERTURBATION_RATIOS for scr in SEARCH_COVERAGE_RATIOS]

metrics_list = [
    {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "dynamic_quantization_iterations": [],
        "dynamic_kernel_reshaping_iterations": [],
        "evaluation_scores": []
    } for _ in range(len(PERTURBATION_RATIOS) * len(SEARCH_COVERAGE_RATIOS))
]

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------- RANDOM SAMPLING NEIGHBORS GENERATION -------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for p_idx, perturbation_ratio in enumerate(PERTURBATION_RATIOS):
    for s_idx, search_coverage_ratio in enumerate(SEARCH_COVERAGE_RATIOS):
        idx = p_idx * len(SEARCH_COVERAGE_RATIOS) + s_idx
        for run in range(RUNS):
            print(f"\n--- TEST NAME: {TEST_NAME} \t Random Sampling Neighbors Generation \t Perturbation Ratio: {perturbation_ratio}, Search Coverage Ratio: {search_coverage_ratio} \t Vanilla ASTAR Training Run {run + 1} ---\n")
            
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
                                    perturbation_ratio=perturbation_ratio, search_coverage_ratio=search_coverage_ratio,
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

            metrics_list[idx]["losses"].append(trainer.loss_history)
            metrics_list[idx]["training_times"].append(trainer.training_time)
            metrics_list[idx]["final_losses"].append(trainer.best_node.h_val)
            metrics_list[idx]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
            metrics_list[idx]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

all_results = {label: metric for label, metric in zip(labels_list, metrics_list)}

save_metrics(all_results, TEST_NAME)

generate_evaluation_statistical_summary(metrics_list,labels_list, TEST_NAME)

generate_plots(metrics_list, labels_list, TEST_NAME, DATASET_NAME)

plot_regression_statistics(metrics_list, labels_list, TEST_NAME, DATASET_NAME)
