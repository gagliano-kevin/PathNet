import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import torch
from sklearn.model_selection import train_test_split
import numpy as np


def plot_circles_dataset(X, y, noise_level=0.1, factor_level=0.5, n_samples=1000):
    """
    Generate and plot a synthetic circles dataset.
    Parameters:
    - noise_level: float, standard deviation of Gaussian noise added to the data.
    - factor_level: float, scale factor between inner and outer circle.
    - n_samples: int, total number of samples to generate.
    """
    plt.figure(figsize=(6, 6))
    # Create a scatter plot where the color 'c' is determined by the label 'y'.
    # The 'coolwarm' colormap is useful for binary classification.
    plt.scatter(
        X[:, 0], # Feature 1 (X-axis)
        X[:, 1], # Feature 2 (Y-axis)
        c=y, 
        cmap=plt.cm.coolwarm,
        edgecolor='k', # Black border around points
        s=40 # Size of points
    )
    plt.title(f"Synthetic Circles Dataset (Noise={noise_level}, Factor={factor_level})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    #plt.show() 
    plt.savefig("circles_dataset.png") 


def get_circle_data_tensors(n_samples=1000, noise_level=0.1, factor_level=0.5, random_seed=42):
    """Generates and splits the circle dataset, returning PyTorch tensors."""
    X, y = make_circles(
        n_samples=n_samples,
        noise=noise_level, 
        factor=factor_level,
        random_state=random_seed
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
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

def pad_losses(losses_list, target_len):
    """Pads all loss histories in the list up to the target_len with NaN."""
    padded_array = np.full((len(losses_list), target_len), np.nan)
    for i, l in enumerate(losses_list):
        padded_array[i, :len(l)] = l
    return padded_array


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
