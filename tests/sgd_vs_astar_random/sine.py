#==============================================================================================================================================================
#==============================================================================================================================================================
#-------------------------------------------------- python -m tests.sgd_vs_astar_random.sine ----------------------------------------------------------
#==============================================================================================================================================================
#==============================================================================================================================================================

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import time

from source.PathNet import TrainerRandomSampling
from source.utils.dataset_utils.sine_utils import generate_sinusoidal_tensor, plot_sine_predictions, SineDataset
from source.utils.evaluation_utils import evaluate_pathnet_regression, evaluate_sgd_regression
from source.utils.plot_utils import generate_plots, save_metrics, generate_evaluation_statistical_summary, plot_regression_statistics
from source.utils.models import SinusoidalMLP

ITERATIONS = 2000
RUNS = 10

MODEL_NAME_PREFIX = "sine_model"
DATASET_NAME = "Noisy Sine Function"

SAVE_TRAINED_MODEL = False
DELTA_ABS = None
EARLY_STOPPING = False

E_S_PATIENCE = 200
LOSS_IMPROVEMENT_THRESHOLD = 1e-3
PARAMETER_RANGE = (-10, 10)
QUANTIZATION_FACTOR = 10

# Parameters for synthetic Sine Dataset
TRAINING_SAMPLES = 1000
VALIDATION_SAMPLES = 1000
TEST_SAMPLES = 1000
MIN_ANGLE = 0
MAX_ANGLE = 2 * np.pi
NOISE_LEVEL = 0.1

INPUT_SIZE = 1
OUTPUT_SIZE = 1

EARLY_STOPPING = False
E_S_PATIENCE = 200
LOSS_IMPROVEMENT_THRESHOLD = 1e-3
PARAMETER_RANGE = (-10, 10)

# SGD Settings
BATCH_SIZE = None    # Full batch
LEARNING_RATE = 0.001
EPOCHS = ITERATIONS
                       
X_train, Y_train = generate_sinusoidal_tensor(num_samples=TRAINING_SAMPLES, min_angle=MIN_ANGLE, max_angle=MAX_ANGLE, noise_level=NOISE_LEVEL)
X_val, Y_val = generate_sinusoidal_tensor(num_samples=VALIDATION_SAMPLES, min_angle=MIN_ANGLE, max_angle=MAX_ANGLE, noise_level=NOISE_LEVEL)
X_test, Y_test = generate_sinusoidal_tensor(num_samples=TEST_SAMPLES, min_angle=MIN_ANGLE, max_angle=MAX_ANGLE, noise_level=NOISE_LEVEL)

print(f"\nTraining Data Shape: {X_train.shape}, {Y_train.shape}")
print(f"Validation Data Shape: {X_val.shape}, {Y_val.shape}")
print(f"Testing Data Shape: {X_test.shape}, {Y_test.shape}\n")

train_dataset = SineDataset(num_samples=TRAINING_SAMPLES, min_angle=MIN_ANGLE, max_angle=MAX_ANGLE, noise_level=NOISE_LEVEL)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = SineDataset(num_samples=TEST_SAMPLES, min_angle=MIN_ANGLE, max_angle=MAX_ANGLE, noise_level=NOISE_LEVEL)
test_dataloader = DataLoader(test_dataset, batch_size=TEST_SAMPLES, shuffle=True)

labels_list = ["A-star", "SGD"]


# =========================================================================================================================================================
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------- SMALL NET --------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================================================================================================================

TEST_NAME = "small_net_sine_SGD_vs_A-star"

# Neural Network Settings
HIDDEN_SIZE_1 = 32
HIDDEN_SIZE_2 = 16

# Parameters for Random Sampling Neighbors Generation
PERTURBATION_RATIO = 0.01       # 1% of the parameters will be perturbed per each neighbor
SEARCH_COVERAGE_RATIO = 0.1     # 10% of the total number of parameters in the model will be the number of neighbors generated per each state

astar_metrics = {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "dynamic_quantization_iterations": [],
        "evaluation_scores": []
    }

grad_metrics = {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "evaluation_scores": []
    }
                       
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------- ASTAR TRAINING ----------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Random Sampling ASTAR Training Run {run + 1} ---\n")

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

    astar_metrics["losses"].append(trainer.loss_history)
    astar_metrics["training_times"].append(trainer.training_time)
    astar_metrics["final_losses"].append(trainer.best_node.h_val)
    astar_metrics["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    astar_metrics["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))

    plot_sine_predictions(test_x_np=X_test.numpy(), 
                          predicted_sin_np=trainer.best_node.quantized_mlp.model(X_test).detach().numpy(), 
                          true_sin_np=Y_test.numpy(),
                          directory=TEST_NAME,
                          filename=f"astar_sine_predictions_run_{run + 1}.png")


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------- GRADIENT BASE TRAINING ---------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    sine_model = SinusoidalMLP(hidden_layers=[HIDDEN_SIZE_1, HIDDEN_SIZE_2])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(sine_model.parameters(), lr=LEARNING_RATE)

    loss_history = []

    start_time = time.perf_counter()

    print(f"\n--- Gradient Training Run {run + 1} ---\n")

    for epoch in range(EPOCHS):
        total_loss = 0
        for x_batch, y_batch in train_dataloader:        
            optimizer.zero_grad()
            predictions = sine_model(x_batch)
            loss = criterion(predictions, y_batch)              
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_history.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_dataloader):.6f}')

    end_time = time.perf_counter()
    training_time = end_time - start_time

    grad_metrics["losses"].append(loss_history)
    grad_metrics["training_times"].append(training_time)
    grad_metrics["final_losses"].append(loss_history[-1])
    grad_metrics["evaluation_scores"].append(evaluate_sgd_regression(sine_model, test_dataloader))

    plot_sine_predictions(test_x_np=test_dataset.x_data.numpy(), 
                predicted_sin_np=sine_model(test_dataset.x_data).detach().numpy(), 
                true_sin_np=test_dataset.sin_y_data.numpy(),
                directory=TEST_NAME,
                filename=f"grad_sine_predictions_run_{run + 1}.png")
    
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

metrics_list = [astar_metrics, grad_metrics]

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

TEST_NAME = "medium_net_sine_SGD_vs_A-star"

# Neural Network Settings
HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 32
HIDDEN_SIZE_3 = 16

# Parameters for Random Sampling Neighbors Generation
PERTURBATION_RATIO = 0.01        # 1% of the parameters will be perturbed per each neighbor
SEARCH_COVERAGE_RATIO = 0.05     # 5% of the total number of parameters in the model will be the number of neighbors generated per each state


astar_metrics = {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "dynamic_quantization_iterations": [],
        "evaluation_scores": []
    }

grad_metrics = {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "evaluation_scores": []
    }
                       
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------- ASTAR TRAINING ----------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Random Sampling ASTAR Training Run {run + 1} ---\n")

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

    astar_metrics["losses"].append(trainer.loss_history)
    astar_metrics["training_times"].append(trainer.training_time)
    astar_metrics["final_losses"].append(trainer.best_node.h_val)
    astar_metrics["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    astar_metrics["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))

    plot_sine_predictions(test_x_np=X_test.numpy(), 
                          predicted_sin_np=trainer.best_node.quantized_mlp.model(X_test).detach().numpy(), 
                          true_sin_np=Y_test.numpy(),
                          directory=TEST_NAME,
                          filename=f"astar_sine_predictions_run_{run + 1}.png")

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------- GRADIENT BASE TRAINING ---------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    sine_model = SinusoidalMLP(hidden_layers=[HIDDEN_SIZE_1, HIDDEN_SIZE_2, HIDDEN_SIZE_3])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(sine_model.parameters(), lr=LEARNING_RATE)

    loss_history = []

    start_time = time.perf_counter()

    print(f"\n--- Gradient Training Run {run + 1} ---\n")

    for epoch in range(EPOCHS):
        total_loss = 0
        for x_batch, y_batch in train_dataloader:        
            optimizer.zero_grad()
            predictions = sine_model(x_batch)
            loss = criterion(predictions, y_batch)              
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_history.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_dataloader):.6f}')

    end_time = time.perf_counter()
    training_time = end_time - start_time

    grad_metrics["losses"].append(loss_history)
    grad_metrics["training_times"].append(training_time)
    grad_metrics["final_losses"].append(loss_history[-1])
    grad_metrics["evaluation_scores"].append(evaluate_sgd_regression(sine_model, test_dataloader))

    plot_sine_predictions(test_x_np=test_dataset.x_data.numpy(), 
                predicted_sin_np=sine_model(test_dataset.x_data).detach().numpy(), 
                true_sin_np=test_dataset.sin_y_data.numpy(),
                directory=TEST_NAME,
                filename=f"grad_sine_predictions_run_{run + 1}.png")
    
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

metrics_list = [astar_metrics, grad_metrics]

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

TEST_NAME = "big_net_sine_SGD_vs_A-star"

# Neural Network Settings
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64
HIDDEN_SIZE_3 = 32
HIDDEN_SIZE_4 = 16

# Parameters for Random Sampling Neighbors Generation
PERTURBATION_RATIO = 0.1         # 10% of the parameters will be perturbed per each neighbor
SEARCH_COVERAGE_RATIO = 0.05     # 5% of the total number of parameters in the model will be the number of neighbors generated per each state

astar_metrics = {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "dynamic_quantization_iterations": [],
        "evaluation_scores": []
    }

grad_metrics = {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "evaluation_scores": []
    }
                       
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------- ASTAR TRAINING ----------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Random Sampling ASTAR Training Run {run + 1} ---\n")

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

    astar_metrics["losses"].append(trainer.loss_history)
    astar_metrics["training_times"].append(trainer.training_time)
    astar_metrics["final_losses"].append(trainer.best_node.h_val)
    astar_metrics["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    astar_metrics["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))

    plot_sine_predictions(test_x_np=X_test.numpy(), 
                          predicted_sin_np=trainer.best_node.quantized_mlp.model(X_test).detach().numpy(), 
                          true_sin_np=Y_test.numpy(),
                          directory=TEST_NAME,
                          filename=f"astar_sine_predictions_run_{run + 1}.png")

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------- GRADIENT BASE TRAINING ---------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    sine_model = SinusoidalMLP(hidden_layers=[HIDDEN_SIZE_1, HIDDEN_SIZE_2, HIDDEN_SIZE_3, HIDDEN_SIZE_4])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(sine_model.parameters(), lr=LEARNING_RATE)

    loss_history = []

    start_time = time.perf_counter()

    print(f"\n--- Gradient Training Run {run + 1} ---\n")

    for epoch in range(EPOCHS):
        total_loss = 0
        for x_batch, y_batch in train_dataloader:        
            optimizer.zero_grad()
            predictions = sine_model(x_batch)
            loss = criterion(predictions, y_batch)              
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_history.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_dataloader):.6f}')

    end_time = time.perf_counter()
    training_time = end_time - start_time

    grad_metrics["losses"].append(loss_history)
    grad_metrics["training_times"].append(training_time)
    grad_metrics["final_losses"].append(loss_history[-1])
    grad_metrics["evaluation_scores"].append(evaluate_sgd_regression(sine_model, test_dataloader))

    plot_sine_predictions(test_x_np=test_dataset.x_data.numpy(), 
                predicted_sin_np=sine_model(test_dataset.x_data).detach().numpy(), 
                true_sin_np=test_dataset.sin_y_data.numpy(),
                directory=TEST_NAME,
                filename=f"grad_sine_predictions_run_{run + 1}.png")
    
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

metrics_list = [astar_metrics, grad_metrics]

all_results = {label: metric for label, metric in zip(labels_list, metrics_list)}

save_metrics(all_results, TEST_NAME)

generate_evaluation_statistical_summary(metrics_list,labels_list, TEST_NAME)

generate_plots(metrics_list, labels_list, TEST_NAME, DATASET_NAME)

plot_regression_statistics(metrics_list, labels_list, TEST_NAME, DATASET_NAME)
