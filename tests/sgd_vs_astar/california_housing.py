#==============================================================================================================================================================
#==============================================================================================================================================================
#-------------------------------------------------- python -m tests.sgd_vs_astar.california_housing ----------------------------------------------------------
#==============================================================================================================================================================
#==============================================================================================================================================================

import torch
import torch.nn as nn
import time

from source.PathNet import Trainer
from source.utils.dataset_utils.housing_utils import get_california_housing_data, create_dataloader
from source.utils.evaluation_utils import evaluate_pathnet_regression, evaluate_sgd_regression
from source.utils.plot_utils import generate_plots, save_metrics, generate_evaluation_statistical_summary, plot_regression_statistics
from source.utils.models import HousingMLP

ITERATIONS = 10
RUNS = 1
TEST_NAME = "California Housing - SGD vs A-star"
SAVE_TRAINED_MODEL = False

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

DYNAMIC_KERNEL_RESHAPING = True
D_K_R_PATIENCE = 100
X_WEIGHT_KERNEL_DECR = 1
Y_WEIGHT_KERNEL_DECR = 1
Y_BIAS_KERNEL_DECR = 1
X_STRIDE_DECR = 1
Y_STRIDE_DECR = 1
DELTA_ABS = None

# Neural Network Settings
INPUT_SIZE = 8
HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 32
HIDDEN_SIZE_3 = 16
OUTPUT_SIZE = 1

EARLY_STOPPING = False
E_S_PATIENCE = 200
LOSS_IMPROVEMENT_THRESHOLD = 1e-3
PARAMETER_RANGE = (-10, 10)

# Dynamic Quantization Settings
D_Q_PATIENCE = 100
QUANTIZATION_FACTOR_MULTIPLIER = 10
MIN_QUANTIZATION_FACTOR = 10
MAX_QUANTIZATION_FACTOR = 1e4

astar_metrics = {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "dynamic_quantization_iterations": [],
        "dynamic_kernel_reshaping_iterations": [],
        "evaluation_scores": []
    }

grad_metrics = {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "evaluation_scores": []
    }
                       
X_train, Y_train, X_val, Y_val, X_test, Y_test = get_california_housing_data()
print(f"\nTraining Data Shape: {X_train.shape}, {Y_train.shape}")
print(f"Validation Data Shape: {X_val.shape}, {Y_val.shape}")
print(f"Testing Data Shape: {X_test.shape}, {Y_test.shape}\n")


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------- ASTAR TRAINING ----------------------------------------------------------------
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
                            quantization_factor=MIN_QUANTIZATION_FACTOR,
                            parameter_range=PARAMETER_RANGE,
                            debug_mlp=False,
                            #----------------------------------------------------------------------------------
                            weight_kernel = MAX_WEIGHT_KERNEL, bias_kernel = MAX_BIAS_KERNEL, x_stride=MAX_X_STRIDE, y_stride=MAX_Y_STRIDE, delta_abs=DELTA_ABS,
                            #----------------------------------------------------------------------------------
                            early_stopping=EARLY_STOPPING, e_s_patience=E_S_PATIENCE,
                            #----------------------------------------------------------------------------------
                            dynamic_quantization=True, d_q_patience=D_Q_PATIENCE, 
                            quantization_factor_multiplier=QUANTIZATION_FACTOR_MULTIPLIER, max_quantization_factor=MAX_QUANTIZATION_FACTOR,
                            #-----------------------------------------------------------------------------------
                            dynamic_kernel_reshaping=True, d_k_r_patience=D_K_R_PATIENCE, 
                            x_weight_kernel_decr=X_WEIGHT_KERNEL_DECR, y_weight_kernel_decr=Y_WEIGHT_KERNEL_DECR, y_bias_kernel_decr=Y_BIAS_KERNEL_DECR, 
                            min_weight_kernel=MIN_WEIGHT_KERNEL, min_bias_kernel=MIN_BIAS_KERNEL,
                            x_stride_decr=X_STRIDE_DECR, y_stride_decr=Y_STRIDE_DECR, min_x_stride=MIN_X_STRIDE, min_y_stride=MIN_Y_STRIDE,
                            #----------------------------------------------------------------------------------
                            loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                            #----------------------------------------------------------------------------------
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=f'california_housing_astar_run_{run + 1}'
                            )

    trainer.train(X_train, Y_train)

    astar_metrics["losses"].append(trainer.loss_history)
    astar_metrics["training_times"].append(trainer.training_time)
    astar_metrics["final_losses"].append(trainer.best_node.h_val)
    astar_metrics["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    astar_metrics["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])
    astar_metrics["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))



#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------- GRADIENT BASE TRAINING ---------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

BATCH_SIZE = None    # Full batch
LEARNING_RATE = 0.001
EPOCHS = ITERATIONS

train_dataloader = create_dataloader(X_train, Y_train, batch_size=BATCH_SIZE)

for run in range(RUNS):
    
    housing_model = HousingMLP(input_size=INPUT_SIZE, hidden_size_1=HIDDEN_SIZE_1, hidden_size_2=HIDDEN_SIZE_2, hidden_size_3=HIDDEN_SIZE_3, output_size=OUTPUT_SIZE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(housing_model.parameters(), lr=LEARNING_RATE)

    loss_history = []

    start_time = time.perf_counter()

    print(f"\n--- Gradient Training Run {run + 1} ---\n")

    for epoch in range(EPOCHS):
        total_loss = 0
        for x_batch, y_batch in train_dataloader:        
            optimizer.zero_grad()
            predictions = housing_model(x_batch)
            loss = criterion(predictions, y_batch)              
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_history.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Housing Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_dataloader):.6f}')


    end_time = time.perf_counter()
    training_time = end_time - start_time

    test_dataloader = create_dataloader(X_test, Y_test, batch_size=BATCH_SIZE)

    grad_metrics["losses"].append(loss_history)
    grad_metrics["training_times"].append(training_time)
    grad_metrics["final_losses"].append(loss_history[-1])
    grad_metrics["evaluation_scores"].append(evaluate_sgd_regression(housing_model, test_dataloader))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

metrics_list = [astar_metrics, grad_metrics]
labels_list = ["A-star", "SGD"]
DATASET_NAME = "California Housing"

generate_evaluation_statistical_summary(metrics_list,labels_list, TEST_NAME)

generate_plots(metrics_list, labels_list, TEST_NAME, DATASET_NAME)

plot_regression_statistics(metrics_list, labels_list, TEST_NAME, DATASET_NAME)

all_results = {label: metric for label, metric in zip(labels_list, metrics_list)}

save_metrics(all_results, TEST_NAME)
