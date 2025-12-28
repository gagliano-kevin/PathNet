import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class WineMLP(nn.Module):
    def __init__(self, input_size=11, hidden_size_1=32, hidden_size_2=32, output_size=6):
        super(WineMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, output_size)
        )
        
    def forward(self, x):
        return self.net(x)



class HousingMLP(torch.nn.Module):
    def __init__(self, input_size=8, hidden_size_1=32, hidden_size_2=32, output_size=1):
        super(HousingMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, output_size)
        )

    def forward(self, x):
        return self.net(x)



class IrisMLP(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, num_classes):
        super().__init__()
        # 4 features -> Hidden Layer 1
        self.fc1 = nn.Linear(input_size, hidden_size_1) 
        # Hidden Layer 1 -> Hidden Layer 2
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        # Hidden Layer 2 -> Hidden Layer 3
        self.fc3 = nn.Linear(hidden_size_2, hidden_size_3)
        # Hidden Layer 3 -> 3 classes
        self.fc4 = nn.Linear(hidden_size_3, num_classes)
        
    def forward(self, x):
        # ReLU activation for hidden layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        # No activation on the output layer when using nn.CrossEntropyLoss
        # (it handles the softmax internally for better numerical stability)
        out = self.fc4(x)
        return out



class SinusoidalMLP(nn.Module):
    def __init__(self, input_size=1, hidden_size_1=32, hidden_size_2=32, hidden_size_3=32, output_size=1):
        super(SinusoidalMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, hidden_size_3),
            nn.ReLU(),
            nn.Linear(hidden_size_3, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)
    

# used in: optimizations_comparison.sine_comparison
def plot_losses(loss_lists, labels, filename="loss_plot.png"):
    """
    Plots multiple lists of loss values on a single figure, regardless of length.

    Args:
        loss_lists (list of lists/arrays): The loss values for each training run.
        labels (list of str): Names for the legend of each run.
    """
    plt.figure(figsize=(10, 6))

    for i, loss_values in enumerate(loss_lists):
        iterations = range(1, len(loss_values) + 1)

        # Plot the loss data
        plt.plot(iterations, loss_values, label=labels[i], alpha=0.8)

    plt.title('Comparison of Training Loss Histories')
    plt.xlabel('Iteration / Epoch Number')
    plt.ylabel('Loss Value')
    plt.legend(title='Type of Training')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()      # Adjusts plot to prevent labels from overlapping
    plt.savefig(filename)
    print(f"Training plot saved in file: {filename}")


def pad_losses(losses_list, target_len):
    """Pads all loss histories in the list up to the target_len with NaN."""
    padded_array = np.full((len(losses_list), target_len), np.nan)
    for i, l in enumerate(losses_list):
        padded_array[i, :len(l)] = l
    return padded_array


def plot_mean_loss_with_std(astar_mean, astar_std, grad_mean, grad_std, runs, filename="mean_loss_comparison_with_std.png", dataset_name="dataset"):
    """Plots the mean loss over epochs/iterations with a shaded region for standard deviation."""
    
    # epochs is now correctly determined by the global maximum length
    epochs = np.arange(len(astar_mean)) + 1
    
    plt.figure(figsize=(10, 6))

    # Plot A-Star 
    plt.plot(epochs, astar_mean, label='A-Star (Mean Loss)', color='blue')
    plt.fill_between(epochs, astar_mean - astar_std, astar_mean + astar_std, 
                     alpha=0.2, color='blue', label='A-Star ($\pm 1 \sigma$)')

    # Plot Gradient Descent 
    plt.plot(epochs, grad_mean, label='Gradient Descent (Mean Loss)', color='red')
    plt.fill_between(epochs, grad_mean - grad_std, grad_mean + grad_std, 
                     alpha=0.2, color='red', label='Gradient Descent ($\pm 1 \sigma$)')

    plt.title(f'Mean Training Loss Comparison on {dataset_name} over {runs} Runs')
    plt.xlabel('Epochs / Iterations')
    plt.ylabel('Mean Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")


def plot_final_loss_distribution(astar_final_losses, grad_final_losses, runs, filename="final_loss_boxplot.png", dataset_name="dataset"):
    """Plots a Box-and-Whisker plot of the final performance metric."""
    
    data = [astar_final_losses, grad_final_losses]
    labels = ['A-Star', 'Gradient Descent']
    
    plt.figure(figsize=(8, 6))
    
    # Boxplot showing median, IQR, and range
    plt.boxplot(data, vert=True, patch_artist=True, labels=labels, 
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='darkred'))
    
    # Add individual points (jitter) to show all run results
    for i, losses in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(losses)) 
        plt.scatter(x, losses, color='black', alpha=0.6, s=10)

    plt.title(f'Distribution of Final Loss on {dataset_name} over {runs} Runs')
    plt.ylabel('Final Loss')
    plt.xticks(ticks=[1, 2], labels=labels)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")


def plot_mean_loss_with_std(labels, static_astar_mean, static_astar_std, dynamic_astar_mean, dynamic_astar_std, runs, filename="mean_loss_comparison_with_std.png", dataset_name="dataset"):
    """Plots the mean loss over epochs/iterations with a shaded region for standard deviation."""
    
    # epochs is now correctly determined by the global maximum length
    epochs = np.arange(len(static_astar_mean)) + 1
    
    plt.figure(figsize=(10, 6))

    # Plot A-Star 
    plt.plot(epochs, static_astar_mean, label=f'{labels[0]} (Mean Loss)', color='blue')
    plt.fill_between(epochs, static_astar_mean - static_astar_std, static_astar_mean + static_astar_std, 
                     alpha=0.2, color='blue', label=f'{labels[0]} ($\pm 1 \sigma$)')

    # pad with nan the shorter one if lengths differ
    len_static = len(static_astar_mean)
    len_dynamic = len(dynamic_astar_mean)
    if len_static < len_dynamic:
        static_astar_mean = np.pad(static_astar_mean, (0, len_dynamic - len_static), constant_values=np.nan)
        static_astar_std = np.pad(static_astar_std, (0, len_dynamic - len_static), constant_values=np.nan)
    elif len_dynamic < len_static:
        dynamic_astar_mean = np.pad(dynamic_astar_mean, (0, len_static - len_dynamic), constant_values=np.nan)
        dynamic_astar_std = np.pad(dynamic_astar_std, (0, len_static - len_dynamic), constant_values=np.nan)

    plt.plot(epochs, dynamic_astar_mean, label=f'{labels[1]}  (Mean Loss)', color='red')
    plt.fill_between(epochs, dynamic_astar_mean - dynamic_astar_std, dynamic_astar_mean + dynamic_astar_std, 
                     alpha=0.2, color='red', label=f'{labels[1]} ($\pm 1 \sigma$)')

    plt.title(f'Mean Training Loss Comparison on {dataset_name} over {runs} Runs')
    plt.xlabel('Epochs / Iterations')
    plt.ylabel('Mean Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")


def plot_final_loss_distribution(labels, static_astar_final_losses, dynamic_astar_final_losses, runs, filename="final_loss_boxplot.png", dataset_name="dataset"):
    """Plots a Box-and-Whisker plot of the final performance metric."""
    
    data = [static_astar_final_losses, dynamic_astar_final_losses]
    
    plt.figure(figsize=(8, 6))
    
    # Boxplot showing median, IQR, and range
    plt.boxplot(data, vert=True, patch_artist=True, labels=labels, 
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='darkred'))
    
    # Add individual points (jitter) to show all run results
    for i, losses in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(losses)) 
        plt.scatter(x, losses, color='black', alpha=0.6, s=10)

    plt.title(f'Distribution of Final Loss on {dataset_name} over {runs} Runs')
    plt.ylabel('Final Loss')
    plt.xticks(ticks=[1, 2], labels=labels)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")