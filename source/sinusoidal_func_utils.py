import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class SinCosDataset(Dataset):
    def __init__(self, num_samples, min_angle, max_angle, noise_level=0.1):
        self.x_data = torch.linspace(min_angle, max_angle, num_samples).float().unsqueeze(1)
        
        sin_y = torch.sin(self.x_data)
        cos_y = torch.cos(self.x_data)
        
        # Add noise
        noise = torch.randn_like(sin_y) * noise_level
        self.sin_y_data = sin_y + noise
        self.cos_y_data = cos_y + noise

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        # Returns x, sin(x), cos(x)
        return self.x_data[idx], self.sin_y_data[idx], self.cos_y_data[idx]
    


class SinusoidalMLP(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(SinusoidalMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)



def generate_sinusoidal_tensor(func, num_samples, min_angle, max_angle, noise_level):
    """Generates X and Y tensors for a given function (torch.sin or torch.cos)."""
    # X tensor (angles)
    X = torch.linspace(min_angle, max_angle, num_samples).float().unsqueeze(1)
    
    # Y tensor (function value + noise)
    Y_true = func(X)
    noise = torch.randn_like(Y_true) * noise_level
    Y = Y_true + noise
    
    return X, Y



def plot_sine_data(X: torch.Tensor, Y: torch.Tensor, filename="sine_data.png"):
    """Plots the noisy sine data."""
    plt.figure(figsize=(7, 5))
    plt.scatter(X.numpy(), Y.numpy(), s=10, color='blue', alpha=0.6, label='Noisy Data')
    plt.title('Noisy Sine Data')
    plt.xlabel('Angle (x)')
    plt.ylabel('sin(x) with noise')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(filename)


    
def plot_sine_predictions(test_x_np: np.ndarray, 
                          predicted_sin_np: np.ndarray, 
                          true_sin_np: np.ndarray, 
                          filename = "sine_plot.png"):
    """
    Plots the predicted sine values against the true sine values.

    Args:
        test_x_np (np.ndarray): The input angles (x-axis data).
        predicted_sin_np (np.ndarray): The network's predicted sin(x) values.
        true_sin_np (np.ndarray): The actual sin(x) values.
    """
    
    plt.figure(figsize=(7, 5))
    
    # Plotting for Sine function
    plt.plot(test_x_np, true_sin_np, label='True $\\sin(x)$', color='blue', linewidth=2)
    plt.plot(test_x_np, predicted_sin_np, '--', label='Predicted $\\sin(x)$', color='red', linewidth=1.5, alpha=0.8)
    
    plt.title('Sine Function Regression Prediction')
    plt.xlabel('Angle (x)')
    plt.ylabel('$\\sin(x)$')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(filename)


def plot_mean_loss_with_std(astar_mean, astar_std, grad_mean, grad_std, iterations, filename="mean_loss_comparison_with_std.png"):
    """Plots the mean loss over epochs/iterations with a shaded region for standard deviation."""
    
    # Create an array of iteration numbers
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

    plt.title(f'Mean Training Loss Comparison over {RUNS} Runs')
    plt.xlabel('Epochs / Iterations')
    plt.ylabel('Mean MSE Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")


def plot_final_loss_distribution(astar_final_losses, grad_final_losses, filename="final_loss_boxplot.png"):
    """Plots a Box-and-Whisker plot of the final performance metric."""
    
    data = [astar_final_losses, grad_final_losses]
    labels = ['A-Star (Novel)', 'Gradient Descent (Classic)']
    
    plt.figure(figsize=(8, 6))
    
    # FIX: medianprops must be a top-level keyword argument, separate from boxprops.
    plt.boxplot(data, vert=True, patch_artist=True, labels=labels, 
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='darkred'))
    
    # Add individual points (jitter) to show all run results
    for i, losses in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(losses)) 
        plt.scatter(x, losses, color='black', alpha=0.6, s=10)

    plt.title(f'Distribution of Final Loss over {RUNS} Runs')
    plt.ylabel('Final MSE Loss')
    plt.xticks(ticks=[1, 2], labels=labels)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")