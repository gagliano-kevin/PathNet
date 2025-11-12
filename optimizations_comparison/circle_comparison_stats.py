#===================================================================================================================================
#===================================================================================================================================
#----------------- run this file from project root: python -m optimizations_comparison.circle_comparison_stats ---------------------
#===================================================================================================================================
#===================================================================================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Assuming these utilities exist in your project structure
# from source.PathNet import Trainer (Assuming PathNet Trainer handles A-Star logic)
from source.SimplePathNet import Trainer # Using SimplePathNet Trainer as fallback if PathNet is unavailable

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import time
import numpy as np
import matplotlib.pyplot as plt


# --- GLOBAL CONFIGURATION ---
RUNS = 10           # Number of times to run the experiment for statistical analysis
MAX_ITERATIONS = 1000 # Number of iterations/epochs for both methods (This value will be overridden if any run exceeds it)
LEARNING_RATE = 0.01

# Dataset parameters
N_SAMPLES = 1000
NOISE_LEVEL = 0.1
FACTOR_LEVEL = 0.5 
RANDOM_SEED = 42

# Model parameters
INPUT_SIZE = 2      # X, Y coordinates
HIDDEN_SIZE = 4
OUTPUT_SIZE = 2     # Two classes (inner/outer circle)

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

LOG_FILE_ASTAR = "circle_model_astar_multiple_runs"
LOG_FILE_GRAD = "circle_model_grad_base_multiple_runs"


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- DATA SETUP ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

def get_circle_data_tensors():
    """Generates and splits the circle dataset, returning PyTorch tensors."""
    X, y = make_circles(
        n_samples=N_SAMPLES,
        noise=NOISE_LEVEL, 
        factor=FACTOR_LEVEL,
        random_state=RANDOM_SEED
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    print("\n--- Data Preparation Summary ---")
    print(f"Input Feature Count (input_size): {X.shape[1]}")
    print(f"Output Class Count (num_classes): {len(np.unique(y))}")
    print(f"Training Set Size: {X_train_tensor.shape[0]}")
    print(f"Test Set Size: {X_test_tensor.shape[0]}\n")
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = get_circle_data_tensors()

# Prepare DataLoader for Gradient Descent (using full batch to align with A-Star iteration count)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=X_train_tensor.shape[0], shuffle=True)


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- ASTAR TRAINING -----------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- ASTAR Training Run {run + 1}/{RUNS} ---\n")

    # Simple neural network model for circle classification (CrossEntropyLoss expects logits)
    model = nn.Sequential(
        nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
    ) 

    trainer = Trainer(model, nn.CrossEntropyLoss(), 
                      quantization_factor=10, 
                      parameter_range=(-10, 10), 
                      debug_mlp=True, 
                      param_fraction=1.0, 
                      max_iterations=MAX_ITERATIONS, 
                      log_freq=50, 
                      target_loss=0.0001, 
                      measure_time=True) 

    trainer.train(X_train_tensor, y_train_tensor)

    # Collect Metrics
    ASTAR_METRICS["losses"].append(trainer.loss_history)
    ASTAR_METRICS["training_times"].append(trainer.training_times[-1]) 
    
    # Using the final heuristic cost (h_val) as the final loss proxy
    final_loss_astar = trainer.best_node.h_val 
    ASTAR_METRICS["final_losses"].append(final_loss_astar)


#------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------- GRADIENT BASE TRAINING ---------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    
    # Model definition
    model = nn.Sequential(
        nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
    ) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_history = []
    
    start_time = time.time()

    print(f"\n--- Gradient Training Run {run + 1}/{RUNS} ---\n")

    model.train() 
    for epoch in range(MAX_ITERATIONS):
        # Full batch training (1 batch per epoch)
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad() 
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch) 
            loss.backward()      
            optimizer.step()     
        
        loss_history.append(loss.item()) 

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{MAX_ITERATIONS}], Loss: {loss.item():.4f}')


    end_time = time.time()
    training_time = end_time - start_time

    # Collect Metrics
    GRAD_METRICS["losses"].append(loss_history)
    GRAD_METRICS["training_times"].append(training_time)
    GRAD_METRICS["final_losses"].append(loss_history[-1])


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- STATISTICAL ANALYSIS & PLOTTING ------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------


# --- ALIGNMENT FIX: DETERMINE GLOBAL MAX LENGTH FOR PLOTTING ---

all_losses = ASTAR_METRICS["losses"] + GRAD_METRICS["losses"]
# Find the maximum length across all runs/methods (e.g., 1000 from the error trace)
global_max_len = max(len(l) for l in all_losses) if all_losses else MAX_ITERATIONS 

def pad_losses(losses_list, target_len):
    """Pads all loss histories in the list up to the target_len with NaN."""
    padded_array = np.full((len(losses_list), target_len), np.nan)
    for i, l in enumerate(losses_list):
        padded_array[i, :len(l)] = l
    return padded_array

# Pad both arrays to the global maximum length (e.g., 1000)
astar_losses_array = pad_losses(ASTAR_METRICS["losses"], global_max_len)
grad_losses_array = pad_losses(GRAD_METRICS["losses"], global_max_len)

# --- CALCULATE MEAN/STD DEV ---

# Calculate mean and standard deviation across all runs for each iteration (now safe due to equal length)
astar_mean_loss = np.nanmean(astar_losses_array, axis=0)
astar_std_loss = np.nanstd(astar_losses_array, axis=0)

grad_mean_loss = np.nanmean(grad_losses_array, axis=0)
grad_std_loss = np.nanstd(grad_losses_array, axis=0)


# --- 1. FINAL LOSS STATS (for Summary Table and Box Plot) ---
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

with open(f"circle_training_statistics_summary_{RUNS}_runs.txt", "w") as f:
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

print(f"\nSaved statistical summary to 'circle_training_statistics_summary_{RUNS}_runs.txt'\n")


#----------------------------------------------------------------------------------------------------------------------------------------------------------
# --- PLOTTING FUNCTIONS (No changes needed here as the input arrays are now correctly aligned) ---

def plot_mean_loss_with_std(astar_mean, astar_std, grad_mean, grad_std, filename="circle_mean_loss_comparison_with_std.png"):
    """Plots the mean loss over epochs/iterations with a shaded region for standard deviation."""
    
    # epochs is now correctly determined by the global maximum length
    epochs = np.arange(len(astar_mean)) + 1
    
    plt.figure(figsize=(10, 6))

    # Plot A-Star (Novel)
    plt.plot(epochs, astar_mean, label='A-Star (Mean Loss)', color='blue')
    plt.fill_between(epochs, astar_mean - astar_std, astar_mean + astar_std, 
                     alpha=0.2, color='blue', label='A-Star ($\pm 1 \sigma$)')

    # Plot Gradient Descent (Classic)
    plt.plot(epochs, grad_mean, label='Gradient Descent (Mean Loss)', color='red')
    plt.fill_between(epochs, grad_mean - grad_std, grad_mean + grad_std, 
                     alpha=0.2, color='red', label='Gradient Descent ($\pm 1 \sigma$)')

    plt.title(f'Mean Training Cross-Entropy Loss Comparison on Circles over {RUNS} Runs')
    plt.xlabel('Epochs / Iterations')
    plt.ylabel('Mean Cross-Entropy Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")


def plot_final_loss_distribution(astar_final_losses, grad_final_losses, filename="circle_final_loss_boxplot.png"):
    """Plots a Box-and-Whisker plot of the final performance metric."""
    
    data = [astar_final_losses, grad_final_losses]
    labels = ['A-Star (Novel)', 'Gradient Descent (Classic)']
    
    plt.figure(figsize=(8, 6))
    
    # Boxplot showing median, IQR, and range
    plt.boxplot(data, vert=True, patch_artist=True, labels=labels, 
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='darkred'))
    
    # Add individual points (jitter) to show all run results
    for i, losses in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(losses)) 
        plt.scatter(x, losses, color='black', alpha=0.6, s=10)

    plt.title(f'Distribution of Final Cross-Entropy Loss on Circles over {RUNS} Runs')
    plt.ylabel('Final Cross-Entropy Loss')
    plt.xticks(ticks=[1, 2], labels=labels)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")


# --- EXECUTE PLOTTING ---

# 1. Plot Mean Loss with Standard Deviation Shading
plot_mean_loss_with_std(astar_mean_loss, astar_std_loss, grad_mean_loss, grad_std_loss)

# 2. Plot Box and Whisker of Final Losses
plot_final_loss_distribution(astar_final_losses, grad_final_losses)