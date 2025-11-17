#===================================================================================================================================
#===================================================================================================================================
#----------------- run this file from project root: python -m optimizations_comparison.sine_comparison_stats -----------------------------
#===================================================================================================================================
#===================================================================================================================================

from source.sinusoidal_func_utils import generate_sinusoidal_tensor, plot_sine_predictions, SinCosDataset, SinusoidalMLP, plot_final_loss_distribution, plot_mean_loss_with_std
from source.general_utils import plot_losses
from source.PathNet import Trainer

import time

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



NUM_SAMPLES = 1000
MIN_ANGLE = 0
MAX_ANGLE = 4 * np.pi
NOISE_LEVEL = 0.1
ITERATIONS = 5000

RUNS = 10

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

LOG_FILE_ASTAR = "sine_model_astar_multiple_runs"
LOG_FILE_GRAD = "sine_model_grad_base_multiple_runs"

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- ASTAR TRAINING -----------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

X_train, Y_train = generate_sinusoidal_tensor(func=torch.sin, num_samples=NUM_SAMPLES, min_angle=MIN_ANGLE, max_angle=MAX_ANGLE, noise_level=NOISE_LEVEL)


for run in range(RUNS):
    print(f"\n--- ASTAR Training Run {run + 1} ---\n")

    model = nn.Sequential(
        nn.Linear(1, 4),  
        nn.ReLU(),
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
        nn.Tanh()
        )

    trainer = Trainer(model, nn.MSELoss(), quantization_factor=10, parameter_range=(-10, 10), debug_mlp=True, param_fraction=1.0, max_iterations=ITERATIONS, log_freq=100, target_loss=0.01, measure_time=True)

    trainer.train(X_train, Y_train)

    ASTAR_METRICS["losses"].append(trainer.loss_history)
    ASTAR_METRICS["training_times"].append(trainer.training_times[-1])
    ASTAR_METRICS["final_losses"].append(trainer.best_node.h_val + trainer.target_loss)

    trainer.log_to_file(f"{LOG_FILE_ASTAR}_run_{run + 1}.txt")

    plot_sine_predictions(test_x_np=X_train.numpy(), 
                          predicted_sin_np=trainer.best_node.quantized_mlp.model(X_train).detach().numpy(), 
                          true_sin_np=Y_train.numpy(),
                          filename=f"{LOG_FILE_ASTAR}_run_{run + 1}.png")


#------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------- GRADIENT BASE TRAINING ---------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------


BATCH_SIZE = NUM_SAMPLES    # Full batch
LEARNING_RATE = 0.001
EPOCHS = ITERATIONS
HIDDEN_SIZE = 4

dataset = SinCosDataset(NUM_SAMPLES, MIN_ANGLE, MAX_ANGLE, NOISE_LEVEL)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for run in range(RUNS):
    
    sin_model_tanh_out = SinusoidalMLP(hidden_size=HIDDEN_SIZE)
    criterion = nn.MSELoss()
    sin_optimizer = torch.optim.Adam(sin_model_tanh_out.parameters(), lr=LEARNING_RATE)

    loss_history = []

    start_time = time.time()

    print(f"\n--- Gradient Training Run {run + 1} ---\n")

    for epoch in range(EPOCHS):
        total_loss = 0
        for x_batch, sin_y_batch, cos_y_batch in dataloader:        # Only using sin_y_batch for sine model
            sin_optimizer.zero_grad()
            predictions = sin_model_tanh_out(x_batch)
            loss = criterion(predictions, sin_y_batch)              # Target is sin_y_batch
            loss.backward()
            sin_optimizer.step()
            total_loss += loss.item()
        loss_history.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Sine Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(dataloader):.6f}')


    end_time = time.time()
    training_time = end_time - start_time

    GRAD_METRICS["losses"].append(loss_history)
    GRAD_METRICS["training_times"].append(training_time)
    GRAD_METRICS["final_losses"].append(loss_history[-1])


    with open(f"{LOG_FILE_GRAD}_run_{run + 1}.txt", "w") as f:
        for i, loss in enumerate(loss_history):
            f.write(f"Iteration {i+1}: Loss = {loss}\n")
        f.write(f"\n\nTotal training time (seconds): {training_time:.2f}\n")

    plot_sine_predictions(test_x_np=dataset.x_data.numpy(), 
                        predicted_sin_np=sin_model_tanh_out(dataset.x_data).detach().numpy(), 
                        true_sin_np=dataset.sin_y_data.numpy(),
                        filename=f"{LOG_FILE_GRAD}_run_{run + 1}.png")
    


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


plot_losses(losses, loss_labels, f"sine_loss_comparison_{ITERATIONS}_iters.png")


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
print("| Metric      | A-Star (Novel) | Gradient Descent (Classic) |")
print("|-------------|----------------|----------------------------|")
print(f"| Average Loss| {astar_avg_loss:.6f}     | {grad_avg_loss:.6f}              |")
print(f"| Median Loss | {astar_median:.6f}     | {grad_median:.6f}              |")
print(f"| Std Dev     | {astar_std_dev:.6f}     | {grad_std_dev:.6f}              |")
print(f"| Variance    | {astar_variance:.6f}     | {grad_variance:.6f}              |")
print(f"| Min Loss    | {astar_min:.6f}     | {grad_min:.6f}              |")
print(f"| Max Loss    | {astar_max:.6f}     | {grad_max:.6f}              |")
print(f"| AVG Training Time | {astar_avg_training_time:.6f} | {grad_avg_training_time:.6f}         |")
print("=========================================================================================")

with open(f"sine_training_statistics_summary_{RUNS}_runs.txt", "w") as f:
    f.write("=========================================================================================\n")
    f.write(f"| STATISTICAL SUMMARY over {RUNS} Runs |\n")
    f.write("=========================================================================================\n")
    f.write("| Metric      | A-Star (Novel) | Gradient Descent (Classic) |\n")
    f.write("|-------------|----------------|----------------------------|\n")
    f.write(f"| Average Loss| {astar_avg_loss:.6f}     | {grad_avg_loss:.6f}              |\n")
    f.write(f"| Median Loss | {astar_median:.6f}     | {grad_median:.6f}              |\n")
    f.write(f"| Std Dev     | {astar_std_dev:.6f}     | {grad_std_dev:.6f}              |\n")
    f.write(f"| Variance    | {astar_variance:.6f}     | {grad_variance:.6f}              |\n")
    f.write(f"| Min Loss    | {astar_min:.6f}     | {grad_min:.6f}              |\n")
    f.write(f"| Max Loss    | {astar_max:.6f}     | {grad_max:.6f}              |\n")
    f.write(f"| AVG Training Time | {astar_avg_training_time:.6f} | {grad_avg_training_time:.6f}         |\n")


    f.write("=========================================================================================\n")

print(f"\nSaved statistical summary to 'sine_training_statistics_summary_{RUNS}_runs.txt'\n")

# Convert list of lists of losses into numpy arrays, padding with NaNs if necessary
# This ensures all runs, even if stopped early, can be aggregated correctly.
def align_and_convert_losses(losses_list):
    max_len = max(len(l) for l in losses_list)
    padded_array = np.full((len(losses_list), max_len), np.nan)
    for i, l in enumerate(losses_list):
        padded_array[i, :len(l)] = l
    return padded_array

astar_losses_array = align_and_convert_losses(ASTAR_METRICS["losses"])
grad_losses_array = align_and_convert_losses(GRAD_METRICS["losses"])

# mean and standard deviation across all runs for each iteration
astar_mean_loss = np.nanmean(astar_losses_array, axis=0)
astar_std_loss = np.nanstd(astar_losses_array, axis=0)

grad_mean_loss = np.nanmean(grad_losses_array, axis=0)
grad_std_loss = np.nanstd(grad_losses_array, axis=0)

# mean loss with standard deviation shading
plot_mean_loss_with_std(astar_mean_loss, astar_std_loss, grad_mean_loss, grad_std_loss, ITERATIONS)

# box and whisker of final losses
plot_final_loss_distribution(astar_final_losses, grad_final_losses)
