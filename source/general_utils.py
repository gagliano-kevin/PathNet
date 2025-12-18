import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

class IrisMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # 4 features -> Hidden Layer 1
        self.fc1 = nn.Linear(input_size, hidden_size) 
        # Hidden Layer 1 -> Hidden Layer 2
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Hidden Layer 2 -> 3 classes
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # ReLU activation for hidden layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # No activation on the output layer when using nn.CrossEntropyLoss
        # (it handles the softmax internally for better numerical stability)
        out = self.fc3(x)
        return out
    

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

    plt.title(f'Mean Training Cross-Entropy Loss Comparison on {dataset_name} over {runs} Runs')
    plt.xlabel('Epochs / Iterations')
    plt.ylabel('Mean Cross-Entropy Loss')
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

    plt.title(f'Distribution of Final Cross-Entropy Loss on {dataset_name} over {runs} Runs')
    plt.ylabel('Final Cross-Entropy Loss')
    plt.xticks(ticks=[1, 2], labels=labels)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")