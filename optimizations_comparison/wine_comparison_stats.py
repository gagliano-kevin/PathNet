#===================================================================================================================================
#===================================================================================================================================
#----------------- run this file from project root: python -m optimizations_comparison.wine_comparison_stats --------------------
#===================================================================================================================================
#===================================================================================================================================

from source.wine_utils import get_wine_data, create_dataloader
from source.general_utils import WineMLP, plot_final_loss_distribution, plot_mean_loss_with_std
from source.PathNet import Trainer

import time

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


ITERATIONS = 1000

RUNS = 10

# MLP Hyperparameters
INPUT_SIZE = 11
HIDDEN_SIZE_1 = 8
HIDDEN_SIZE_2 = 16
HIDDEN_SIZE_3 = 8
OUTPUT_SIZE = 6    # 6 classes (0-5 after label shift)

ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": []
}

GRAD_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": []
}

"""
LOG_FILE_ASTAR = "wine_model_astar_multiple_runs"
LOG_FILE_GRAD = "wine_model_grad_base_multiple_runs"
"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- ASTAR TRAINING -----------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

X_train, Y_train, X_test, Y_test = get_wine_data()

OUTPUT_SIZE = len(torch.unique(Y_train))

print(f"\nTraining Data Shape: {X_train.shape}, {Y_train.shape}")
print(f"Testing Data Shape: {X_test.shape}, {Y_test.shape}\n")

for run in range(RUNS):
    print(f"\n--- ASTAR Training Run {run + 1} ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_3),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_3, OUTPUT_SIZE),
            )

    

    trainer = Trainer(model, loss_fn=nn.CrossEntropyLoss(), quantization_factor=10, parameter_range=(-10, 10), debug_mlp=True, \
            weight_kernel=[2,2], bias_kernel=[2], x_stride=2, y_stride=2, delta_abs=None, max_iterations=ITERATIONS, log_freq=100, \
                measure_time=True, save_trained_model=True, model_name=f"wine_classification_model_run_{run + 1}")


    trainer.train(X_train, Y_train)

    ASTAR_METRICS["losses"].append(trainer.loss_history)
    ASTAR_METRICS["training_times"].append(trainer.training_time)
    ASTAR_METRICS["final_losses"].append(trainer.best_node.h_val)

#    trainer.log_to_txt_file(f"{LOG_FILE_ASTAR}_run_{run + 1}.txt")


#------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------- GRADIENT BASE TRAINING ---------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------


BATCH_SIZE = None    # Full batch
LEARNING_RATE = 0.001
EPOCHS = ITERATIONS

dataloader = create_dataloader(X_train, Y_train, batch_size=BATCH_SIZE)

for run in range(RUNS):
    
    wine_model = WineMLP(input_size=INPUT_SIZE, hidden_size_1=HIDDEN_SIZE_1, hidden_size_2=HIDDEN_SIZE_2, hidden_size_3=HIDDEN_SIZE_3, output_size=OUTPUT_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(wine_model.parameters(), lr=LEARNING_RATE)

    loss_history = []

    start_time = time.perf_counter()

    print(f"\n--- Gradient Training Run {run + 1} ---\n")

    for epoch in range(EPOCHS):
        total_loss = 0
        for x_batch, y_batch in dataloader:        
            optimizer.zero_grad()
            predictions = wine_model(x_batch)
            loss = criterion(predictions, y_batch)              
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_history.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Wine Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(dataloader):.6f}')


    end_time = time.perf_counter()
    training_time = end_time - start_time

    GRAD_METRICS["losses"].append(loss_history)
    GRAD_METRICS["training_times"].append(training_time)
    GRAD_METRICS["final_losses"].append(loss_history[-1])

"""
    with open(f"{LOG_FILE_GRAD}_run_{run + 1}.txt", "w") as f:
        for i, loss in enumerate(loss_history):
            f.write(f"Iteration {i+1}: Loss = {loss}\n")
        f.write(f"\n\nTotal training time (seconds): {training_time:.2f}\n")

"""
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

losses = []
loss_labels = []

for run in range(RUNS):
    losses.append(ASTAR_METRICS["losses"][run])
    losses.append(GRAD_METRICS["losses"][run])
    loss_labels.append(f"A-Star Run {run}")
    loss_labels.append(f"Gradient Base Run {run}")


# FINAL LOSS STATS (for Summary Table and Box Plot)
astar_final_losses = np.array(ASTAR_METRICS["final_losses"])
astar_training_times = np.array(ASTAR_METRICS["training_times"])
grad_final_losses = np.array(GRAD_METRICS["final_losses"])
grad_training_times = np.array(GRAD_METRICS["training_times"])


# A-Star Statistics
astar_avg_loss = np.mean(astar_final_losses)
astar_std_dev = np.std(astar_final_losses)
astar_variance = np.var(astar_final_losses)
astar_median = np.median(astar_final_losses)
astar_min = np.min(astar_final_losses)
astar_max = np.max(astar_final_losses)
astar_avg_training_time = np.mean(astar_training_times)

# Gradient Descent Statistics
grad_avg_loss = np.mean(grad_final_losses)
grad_std_dev = np.std(grad_final_losses)
grad_variance = np.var(grad_final_losses)
grad_median = np.median(grad_final_losses)
grad_min = np.min(grad_final_losses)
grad_max = np.max(grad_final_losses)
grad_avg_training_time = np.mean(grad_training_times)


print("\n=========================================================================================")
print(f"| STATISTICAL SUMMARY over {RUNS} Runs |")
print("=========================================================================================")
print("| Metric      | A-Star  | Gradient Descent  |")
print("|-------------|----------------|----------------------------|")
print(f"| Average Loss| {astar_avg_loss:.6f}     | {grad_avg_loss:.6f}              |")
print(f"| Median Loss | {astar_median:.6f}     | {grad_median:.6f}              |")
print(f"| Std Dev     | {astar_std_dev:.6f}     | {grad_std_dev:.6f}              |")
print(f"| Variance    | {astar_variance:.6f}     | {grad_variance:.6f}              |")
print(f"| Min Loss    | {astar_min:.6f}     | {grad_min:.6f}              |")
print(f"| Max Loss    | {astar_max:.6f}     | {grad_max:.6f}              |")
print(f"| AVG Training Time | {astar_avg_training_time:.6f} | {grad_avg_training_time:.6f}         |")
print("=========================================================================================")

with open(f"wine_training_statistics_summary_{RUNS}_runs.txt", "w") as f:
    f.write("=========================================================================================\n")
    f.write(f"| STATISTICAL SUMMARY over {RUNS} Runs |\n")
    f.write("=========================================================================================\n")
    f.write("| Metric      | A-Star  | Gradient Descent |\n")
    f.write("|-------------|----------------|----------------------------|\n")
    f.write(f"| Average Loss| {astar_avg_loss:.6f}     | {grad_avg_loss:.6f}              |\n")
    f.write(f"| Median Loss | {astar_median:.6f}     | {grad_median:.6f}              |\n")
    f.write(f"| Std Dev     | {astar_std_dev:.6f}     | {grad_std_dev:.6f}              |\n")
    f.write(f"| Variance    | {astar_variance:.6f}     | {grad_variance:.6f}              |\n")
    f.write(f"| Min Loss    | {astar_min:.6f}     | {grad_min:.6f}              |\n")
    f.write(f"| Max Loss    | {astar_max:.6f}     | {grad_max:.6f}              |\n")
    f.write(f"| AVG Training Time | {astar_avg_training_time:.6f} | {grad_avg_training_time:.6f}         |\n")


    f.write("=========================================================================================\n")

print(f"\nSaved statistical summary to 'wine_training_statistics_summary_{RUNS}_runs.txt'\n")



astar_losses_array = ASTAR_METRICS["losses"]
grad_losses_array = GRAD_METRICS["losses"]

# mean and standard deviation across all runs for each iteration
astar_mean_loss = np.mean(astar_losses_array, axis=0)
astar_std_loss = np.std(astar_losses_array, axis=0)

grad_mean_loss = np.mean(grad_losses_array, axis=0)
grad_std_loss = np.std(grad_losses_array, axis=0)

# mean loss with standard deviation shading
plot_mean_loss_with_std(astar_mean_loss, astar_std_loss, grad_mean_loss, grad_std_loss, RUNS, "wine_mean_loss_comparison_with_std.png", "Wine Quality")

# box and whisker of final losses
plot_final_loss_distribution(astar_final_losses, grad_final_losses, RUNS, "wine_final_loss_distribution_comparison.png", "Wine Quality")
