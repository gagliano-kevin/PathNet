import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset



class SineDataset(Dataset):
    
    def __init__(self, num_samples, min_angle, max_angle, noise_level=0.1):
        self.x_data = torch.linspace(min_angle, max_angle, num_samples).float().unsqueeze(1)
        
        sin_y = torch.sin(self.x_data)
        
        # Add noise
        noise = torch.randn_like(sin_y) * noise_level
        self.sin_y_data = sin_y + noise

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        # Returns x, sin(x), cos(x)
        return self.x_data[idx], self.sin_y_data[idx]
    


def generate_sinusoidal_tensor(num_samples, min_angle, max_angle, noise_level):
    """Generates X and Y tensors for the noisy sine function."""
    # X tensor (angles)
    X = torch.linspace(min_angle, max_angle, num_samples).float().unsqueeze(1)
    
    # Y tensor (function value + noise)
    Y_true = torch.sin(X)
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

    

import os
import matplotlib.pyplot as plt
import numpy as np

def plot_sine_predictions(test_x_np: np.ndarray, 
                          predicted_sin_np: np.ndarray, 
                          true_sin_np: np.ndarray, 
                          directory: str = "plots",
                          filename: str = "sine_plot.png"):
    """
    Plots the predicted sine values against the true sine values and saves
    the plot in the specified directory.

    Args:
        test_x_np (np.ndarray): The input angles (x-axis data).
        predicted_sin_np (np.ndarray): The network's predicted sin(x) values.
        true_sin_np (np.ndarray): The actual sin(x) values.
        directory (str): The name of the directory to save the plots.
        filename (str): The name of the output file.
    """
    
    # 1. Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # 2. Construct the full save path
    save_path = os.path.join(directory, filename)
    
    plt.figure(figsize=(7, 5))
    
    # Plotting for Sine function
    plt.plot(test_x_np, true_sin_np, label='True $\\sin(x)$', color='blue', linewidth=2)
    plt.plot(test_x_np, predicted_sin_np, '--', label='Predicted $\\sin(x)$', color='red', linewidth=1.5, alpha=0.8)
    
    plt.title('Sine Function Regression Prediction')
    plt.xlabel('Angle (x)')
    plt.ylabel('$\\sin(x)$')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 3. Save to the new path
    plt.savefig(save_path)
    plt.close() # Good practice to close the figure after saving
    
    print(f"Sine function plot saved in: {save_path}")

    