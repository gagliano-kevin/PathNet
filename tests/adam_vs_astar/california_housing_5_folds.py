#==============================================================================================================================================================
#==============================================================================================================================================================
#---------------------------------------------- python -m tests.adam_vs_astar.california_housing_5_folds ------------------------------------------------------
#==============================================================================================================================================================
#==============================================================================================================================================================

import torch
import torch.nn as nn
import time
from sklearn.model_selection import KFold

from source.PathNet2 import Trainer, TrainerLayerWiseKernel, TrainerRandomSampling
from source.utils.dataset_utils.housing_utils import get_california_housing_data, create_dataloader
from source.utils.evaluation_utils import evaluate_pathnet_regression, evaluate_sgd_regression
from source.utils.plot_utils import generate_plots, save_metrics, generate_evaluation_statistical_summary, plot_regression_statistics
from source.utils.models import HousingMLP

ITERATIONS = 1000
FOLDS = 5

MODEL_NAME_PREFIX = "housing_model"
DATASET_NAME = "California Housing"

SAVE_TRAINED_MODEL = False
DELTA_ABS = None
EARLY_STOPPING = False

E_S_PATIENCE = 200
LOSS_IMPROVEMENT_THRESHOLD = 1e-3
PARAMETER_RANGE = (-10, 10)
QUANTIZATION_FACTOR = 10
BEAM_WIDTH = 1e3

INPUT_SIZE = 8
OUTPUT_SIZE = 1

# SGD Settings
BATCH_SIZE = None    # Full batch
LEARNING_RATE = 0.001
EPOCHS = ITERATIONS

# Fetch data and combine into a full dataset for K-Fold CV
X_train, Y_train, X_val, Y_val, X_test, Y_test = get_california_housing_data()
X_all = torch.cat([X_train, X_val, X_test], dim=0)
Y_all = torch.cat([Y_train, Y_val, Y_test], dim=0)

print(f"\nTotal Combined Data Shape for CV: {X_all.shape}, {Y_all.shape}\n")

# Initialize K-Fold
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
cv_folds = list(kf.split(X_all))

labels_list = ["A-star Single Kernel", "A-star Layer-Wise Kernels", "A-star Random Sampling", "Adam"]

# =========================================================================================================================================================
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------- SMALL NET --------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================================================================================================================

TEST_NAME = "small_net_california_housing_Adam_vs_A-star"

# Neural Network Settings
HIDDEN_SIZE_1 = 32
HIDDEN_SIZE_2 = 16

# Parameter for single Kernel Neighbors Generation
WEIGHT_KERNEL = [2,2]
BIAS_KERNEL = [2]
X_STRIDE = 1
Y_STRIDE = 1

# Parameters for Layer-Wise Kernels Neighbors Generation
WEIGHT_KERNELS = [[2,2], [2,2], [1,2]]
BIAS_KERNELS = [[2], [2], [1]]
WEIGHT_STRIDES = [[1,1], [1,1], [1,1]]      # Format: list of [x_stride, y_stride] per layer
BIAS_STRIDES = [[1], [1], [1]]              # Format: list of [stride] per layer

# Parameters for Random Sampling Neighbors Generation
PERTURBATION_RATIO = 0.01       # 1% of the parameters will be perturbed per each neighbor
SEARCH_COVERAGE_RATIO = 0.1     # 10% of the total number of parameters in the model will be the number of neighbors generated per each state

metrics_list = [
    {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "evaluation_scores": []
    } for _ in range(len(labels_list))
]
                       
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------- SINGLE KERNEL NEIGHBORS GENERATION ------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for fold, (train_idx, test_idx) in enumerate(cv_folds):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Single Kernel Neighbors Generation \t BEAM SEARCH ASTAR Training Fold {fold + 1} ---\n")
    
    X_train_fold, Y_train_fold = X_all[train_idx], Y_all[train_idx]
    X_test_fold, Y_test_fold = X_all[test_idx], Y_all[test_idx]

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
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_single_kernel_astar_fold_{fold + 1}'
                            )

    trainer.beam_search_opt_train(X_train_fold, Y_train_fold, BEAM_WIDTH)

    metrics_list[0]["losses"].append(trainer.loss_history)
    metrics_list[0]["training_times"].append(trainer.training_time)
    metrics_list[0]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[0]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test_fold, Y_test_fold)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------- LAYER-WISE KERNELS NEIGHBORS GENERATION ----------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for fold, (train_idx, test_idx) in enumerate(cv_folds):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Layer-Wise Kernels Neighbors Generation \t BEAM SEARCH ASTAR Training Fold {fold + 1} ---\n")
    
    X_train_fold, Y_train_fold = X_all[train_idx], Y_all[train_idx]
    X_test_fold, Y_test_fold = X_all[test_idx], Y_all[test_idx]
    
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
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_layer_wise_kernels_astar_fold_{fold + 1}'
                            )

    trainer.beam_search_opt_train(X_train_fold, Y_train_fold, BEAM_WIDTH)

    metrics_list[1]["losses"].append(trainer.loss_history)
    metrics_list[1]["training_times"].append(trainer.training_time)
    metrics_list[1]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[1]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test_fold, Y_test_fold)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------- RANDOM SAMPLING NEIGHBORS GENERATION -------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for fold, (train_idx, test_idx) in enumerate(cv_folds):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Random Sampling BEAM SEARCH ASTAR Training Fold {fold + 1} ---\n")
    
    X_train_fold, Y_train_fold = X_all[train_idx], Y_all[train_idx]
    X_test_fold, Y_test_fold = X_all[test_idx], Y_all[test_idx]

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
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_random_sampling_astar_fold_{fold + 1}'
                            )

    trainer.beam_search_opt_train(X_train_fold, Y_train_fold, BEAM_WIDTH)

    metrics_list[2]["losses"].append(trainer.loss_history)
    metrics_list[2]["training_times"].append(trainer.training_time)
    metrics_list[2]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[2]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test_fold, Y_test_fold)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------- GRADIENT BASE TRAINING ---------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for fold, (train_idx, test_idx) in enumerate(cv_folds):
    print(f"\n--- Gradient Training Fold {fold + 1} ---\n")
    
    X_train_fold, Y_train_fold = X_all[train_idx], Y_all[train_idx]
    X_test_fold, Y_test_fold = X_all[test_idx], Y_all[test_idx]

    train_dataloader_fold = create_dataloader(X_train_fold, Y_train_fold, batch_size=BATCH_SIZE)
    test_dataloader_fold = create_dataloader(X_test_fold, Y_test_fold, batch_size=BATCH_SIZE)

    housing_model = HousingMLP(hidden_layers=[HIDDEN_SIZE_1, HIDDEN_SIZE_2])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(housing_model.parameters(), lr=LEARNING_RATE)

    loss_history = []
    start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        total_loss = 0
        for x_batch, y_batch in train_dataloader_fold:        
            optimizer.zero_grad()
            predictions = housing_model(x_batch)
            loss = criterion(predictions, y_batch)              
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_history.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Housing Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_dataloader_fold):.6f}')

    end_time = time.perf_counter()
    training_time = end_time - start_time

    metrics_list[3]["losses"].append(loss_history)
    metrics_list[3]["training_times"].append(training_time)
    metrics_list[3]["final_losses"].append(loss_history[-1])
    metrics_list[3]["evaluation_scores"].append(evaluate_sgd_regression(housing_model, test_dataloader_fold))


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

TEST_NAME = "medium_net_california_housing_Adam_vs_A-star"

# Neural Network Settings
HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 32
HIDDEN_SIZE_3 = 16

# Parameter for single Kernel Neighbors Generation
WEIGHT_KERNEL = [4,4]
BIAS_KERNEL = [4]
X_STRIDE = 3
Y_STRIDE = 3

# Parameters for Layer-Wise Kernels Neighbors Generation
WEIGHT_KERNELS = [[4,4], [4,4], [4,4], [1,4]]
BIAS_KERNELS = [[4], [4], [4], [1]]
WEIGHT_STRIDES = [[3,3], [3,3], [3,3], [3,1]]      # Format: list of [x_stride, y_stride] per layer
BIAS_STRIDES = [[3], [3], [3], [1]]                # Format: list of [stride] per layer

# Parameters for Random Sampling Neighbors Generation
PERTURBATION_RATIO = 0.01        # 1% of the parameters will be perturbed per each neighbor
SEARCH_COVERAGE_RATIO = 0.05     # 5% of the total number of parameters in the model will be the number of neighbors generated per each state

metrics_list = [
    {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "evaluation_scores": []
    } for _ in range(len(labels_list))
]
                       
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------- SINGLE KERNEL NEIGHBORS GENERATION ------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for fold, (train_idx, test_idx) in enumerate(cv_folds):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Single Kernel Neighbors Generation \t Beam Search ASTAR Training Fold {fold + 1} ---\n")

    X_train_fold, Y_train_fold = X_all[train_idx], Y_all[train_idx]
    X_test_fold, Y_test_fold = X_all[test_idx], Y_all[test_idx]

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
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_single_kernel_astar_fold_{fold + 1}'
                            )

    trainer.beam_search_opt_train(X_train_fold, Y_train_fold, BEAM_WIDTH)

    metrics_list[0]["losses"].append(trainer.loss_history)
    metrics_list[0]["training_times"].append(trainer.training_time)
    metrics_list[0]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[0]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test_fold, Y_test_fold)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------- LAYER-WISE KERNELS NEIGHBORS GENERATION ----------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for fold, (train_idx, test_idx) in enumerate(cv_folds):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Layer-Wise Kernels Neighbors Generation \t Beam Search ASTAR Training Fold {fold + 1} ---\n")

    X_train_fold, Y_train_fold = X_all[train_idx], Y_all[train_idx]
    X_test_fold, Y_test_fold = X_all[test_idx], Y_all[test_idx]

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
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_layer_wise_kernels_astar_fold_{fold + 1}'
                            )

    trainer.beam_search_opt_train(X_train_fold, Y_train_fold, BEAM_WIDTH)

    metrics_list[1]["losses"].append(trainer.loss_history)
    metrics_list[1]["training_times"].append(trainer.training_time)
    metrics_list[1]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[1]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test_fold, Y_test_fold)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------- RANDOM SAMPLING NEIGHBORS GENERATION -------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for fold, (train_idx, test_idx) in enumerate(cv_folds):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Random Sampling BEAM SEARCH ASTAR Training Fold {fold + 1} ---\n")

    X_train_fold, Y_train_fold = X_all[train_idx], Y_all[train_idx]
    X_test_fold, Y_test_fold = X_all[test_idx], Y_all[test_idx]

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
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_random_sampling_astar_fold_{fold + 1}'
                            )

    trainer.beam_search_opt_train(X_train_fold, Y_train_fold, BEAM_WIDTH)

    metrics_list[2]["losses"].append(trainer.loss_history)
    metrics_list[2]["training_times"].append(trainer.training_time)
    metrics_list[2]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[2]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test_fold, Y_test_fold)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------- GRADIENT BASE TRAINING ---------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for fold, (train_idx, test_idx) in enumerate(cv_folds):
    print(f"\n--- Gradient Training Fold {fold + 1} ---\n")
    
    X_train_fold, Y_train_fold = X_all[train_idx], Y_all[train_idx]
    X_test_fold, Y_test_fold = X_all[test_idx], Y_all[test_idx]

    train_dataloader_fold = create_dataloader(X_train_fold, Y_train_fold, batch_size=BATCH_SIZE)
    test_dataloader_fold = create_dataloader(X_test_fold, Y_test_fold, batch_size=BATCH_SIZE)

    housing_model = HousingMLP(hidden_layers=[HIDDEN_SIZE_1, HIDDEN_SIZE_2, HIDDEN_SIZE_3])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(housing_model.parameters(), lr=LEARNING_RATE)

    loss_history = []
    start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        total_loss = 0
        for x_batch, y_batch in train_dataloader_fold:        
            optimizer.zero_grad()
            predictions = housing_model(x_batch)
            loss = criterion(predictions, y_batch)              
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_history.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Housing Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_dataloader_fold):.6f}')


    end_time = time.perf_counter()
    training_time = end_time - start_time

    metrics_list[3]["losses"].append(loss_history)
    metrics_list[3]["training_times"].append(training_time)
    metrics_list[3]["final_losses"].append(loss_history[-1])
    metrics_list[3]["evaluation_scores"].append(evaluate_sgd_regression(housing_model, test_dataloader_fold))


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

TEST_NAME = "big_net_california_housing_Adam_vs_A-star"

# Neural Network Settings
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64
HIDDEN_SIZE_3 = 32
HIDDEN_SIZE_4 = 16

# Parameter for single Kernel Neighbors Generation
WEIGHT_KERNEL = [6,6]
BIAS_KERNEL = [6]
X_STRIDE = 5
Y_STRIDE = 5

# Parameters for Layer-Wise Kernels Neighbors Generation
WEIGHT_KERNELS = [[6,2], [6,6], [4,4], [4,4], [1,2]]
BIAS_KERNELS = [[6], [6], [4], [2], [1]]
WEIGHT_STRIDES = [[1,5], [5,5], [3,3], [3,3], [1,1]]      # Format: list of [x_stride, y_stride] per layer
BIAS_STRIDES = [[5], [5], [3], [1], [1]]                  # Format: list of [stride] per layer

# Parameters for Random Sampling Neighbors Generation
PERTURBATION_RATIO = 0.1         # 10% of the parameters will be perturbed per each neighbor
SEARCH_COVERAGE_RATIO = 0.05     # 5% of the total number of parameters in the model will be the number of neighbors generated per each state

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

for fold, (train_idx, test_idx) in enumerate(cv_folds):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Single Kernel Neighbors Generation \t Beam Search ASTAR Training Fold {fold + 1} ---\n")
    
    X_train_fold, Y_train_fold = X_all[train_idx], Y_all[train_idx]
    X_test_fold, Y_test_fold = X_all[test_idx], Y_all[test_idx]

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
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_single_kernel_astar_fold_{fold + 1}'
                            )

    trainer.beam_search_opt_train(X_train_fold, Y_train_fold, BEAM_WIDTH)

    metrics_list[0]["losses"].append(trainer.loss_history)
    metrics_list[0]["training_times"].append(trainer.training_time)
    metrics_list[0]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[0]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test_fold, Y_test_fold)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------- LAYER-WISE KERNELS NEIGHBORS GENERATION ----------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for fold, (train_idx, test_idx) in enumerate(cv_folds):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Layer-Wise Kernels Neighbors Generation \t Beam Search ASTAR Training Fold {fold + 1} ---\n")
    
    X_train_fold, Y_train_fold = X_all[train_idx], Y_all[train_idx]
    X_test_fold, Y_test_fold = X_all[test_idx], Y_all[test_idx]

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
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_layer_wise_kernels_astar_fold_{fold + 1}'
                            )

    trainer.beam_search_opt_train(X_train_fold, Y_train_fold, BEAM_WIDTH)

    metrics_list[1]["losses"].append(trainer.loss_history)
    metrics_list[1]["training_times"].append(trainer.training_time)
    metrics_list[1]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[1]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test_fold, Y_test_fold)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------- RANDOM SAMPLING NEIGHBORS GENERATION -------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for fold, (train_idx, test_idx) in enumerate(cv_folds):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Random Sampling ASTAR Training Fold {fold + 1} ---\n")
    
    X_train_fold, Y_train_fold = X_all[train_idx], Y_all[train_idx]
    X_test_fold, Y_test_fold = X_all[test_idx], Y_all[test_idx]

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
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_random_sampling_astar_fold_{fold + 1}'
                            )

    trainer.beam_search_opt_train(X_train_fold, Y_train_fold, BEAM_WIDTH)

    metrics_list[2]["losses"].append(trainer.loss_history)
    metrics_list[2]["training_times"].append(trainer.training_time)
    metrics_list[2]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[2]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test_fold, Y_test_fold)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------- GRADIENT BASE TRAINING ---------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for fold, (train_idx, test_idx) in enumerate(cv_folds):
    print(f"\n--- Gradient Training Fold {fold + 1} ---\n")
    
    X_train_fold, Y_train_fold = X_all[train_idx], Y_all[train_idx]
    X_test_fold, Y_test_fold = X_all[test_idx], Y_all[test_idx]

    train_dataloader_fold = create_dataloader(X_train_fold, Y_train_fold, batch_size=BATCH_SIZE)
    test_dataloader_fold = create_dataloader(X_test_fold, Y_test_fold, batch_size=BATCH_SIZE)

    housing_model = HousingMLP(hidden_layers=[HIDDEN_SIZE_1, HIDDEN_SIZE_2, HIDDEN_SIZE_3, HIDDEN_SIZE_4])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(housing_model.parameters(), lr=LEARNING_RATE)

    loss_history = []
    start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        total_loss = 0
        for x_batch, y_batch in train_dataloader_fold:        
            optimizer.zero_grad()
            predictions = housing_model(x_batch)
            loss = criterion(predictions, y_batch)              
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_history.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Housing Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_dataloader_fold):.6f}')


    end_time = time.perf_counter()
    training_time = end_time - start_time

    metrics_list[3]["losses"].append(loss_history)
    metrics_list[3]["training_times"].append(training_time)
    metrics_list[3]["final_losses"].append(loss_history[-1])
    metrics_list[3]["evaluation_scores"].append(evaluate_sgd_regression(housing_model, test_dataloader_fold))

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

all_results = {label: metric for label, metric in zip(labels_list, metrics_list)}

save_metrics(all_results, TEST_NAME)
generate_evaluation_statistical_summary(metrics_list,labels_list, TEST_NAME)
generate_plots(metrics_list, labels_list, TEST_NAME, DATASET_NAME)
plot_regression_statistics(metrics_list, labels_list, TEST_NAME, DATASET_NAME)