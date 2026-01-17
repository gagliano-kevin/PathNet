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



