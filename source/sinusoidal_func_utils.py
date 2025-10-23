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
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

# Hyperparameters
NUM_SAMPLES = 1000
MIN_ANGLE = 0
MAX_ANGLE = 4 * np.pi
NOISE_LEVEL = 0.1

def generate_sinusoidal_tensor(func, num_samples, min_angle, max_angle, noise_level):
    """Generates X and Y tensors for a given function (torch.sin or torch.cos)."""
    # X tensor (angles)
    X = torch.linspace(min_angle, max_angle, num_samples).float().unsqueeze(1)
    
    # Y tensor (function value + noise)
    Y_true = func(X)
    noise = torch.randn_like(Y_true) * noise_level
    Y = Y_true + noise
    
    return X, Y



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
# ------------------------------------------------------------------------------

def plot_cosine_predictions(test_x_np: np.ndarray, 
                            predicted_cos_np: np.ndarray, 
                            true_cos_np: np.ndarray):
    """
    Plots the predicted cosine values against the true cosine values.

    Args:
        test_x_np (np.ndarray): The input angles (x-axis data).
        predicted_cos_np (np.ndarray): The network's predicted cos(x) values.
        true_cos_np (np.ndarray): The actual cos(x) values.
    """
    
    plt.figure(figsize=(7, 5))

    # Plotting for Cosine function
    plt.plot(test_x_np, true_cos_np, label='True $\\cos(x)$', color='blue', linewidth=2)
    plt.plot(test_x_np, predicted_cos_np, '--', label='Predicted $\\cos(x)$', color='red', linewidth=1.5, alpha=0.8)
    
    plt.title('Cosine Function Regression Prediction')
    plt.xlabel('Angle (x)')
    plt.ylabel('$\\cos(x)$')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# ==============================================================================
# Example of how to call these methods (assuming you have your data ready)
# ==============================================================================

"""if __name__ == '__main__':
    # 1. Generate dummy data for demonstration
    MIN_ANGLE = 0
    MAX_ANGLE = 4 * np.pi
    TEST_SAMPLES = 500
    
    test_x_np = np.linspace(MIN_ANGLE, MAX_ANGLE, TEST_SAMPLES)
    
    # True values
    true_sin_np = np.sin(test_x_np)
    true_cos_np = np.cos(test_x_np)
    
    # Simulated (predicted) values 
    # (A good model's predictions would be very close to true values)
    predicted_sin_np = true_sin_np + np.random.normal(0, 0.05, TEST_SAMPLES)
    predicted_cos_np = true_cos_np - np.random.normal(0, 0.05, TEST_SAMPLES)

    # 2. Call the separate plotting functions
    print("Plotting Sine Predictions:")
    plot_sine_predictions(test_x_np, predicted_sin_np, true_sin_np)

    print("Plotting Cosine Predictions:")
    plot_cosine_predictions(test_x_np, predicted_cos_np, true_cos_np)"""