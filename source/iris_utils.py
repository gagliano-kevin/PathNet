import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


def get_iris_data_tensors():
    iris = load_iris()
    X, y = iris.data, iris.target

    # Scaling features for better performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor



def print_iris_data_info():
    tensors = get_iris_data_tensors()
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = tensors

    # Info of the dataset
    print("--- Data Variables Ready for Custom Model ---")
    print(f"Input Feature Count (input_size): {X_train_tensor.shape[1]}")
    print(f"Output Class Count (num_classes): {len(np.unique(y_train_tensor.numpy()))}")
    print(f"Training Set Size: {X_train_tensor.shape[0]}")
    print(f"Test Set Size: {X_test_tensor.shape[0]}\n")
    print(f"X_train_tensor shape: {X_train_tensor.shape}\n")
    print(f"y_train_tensor shape: {y_train_tensor.shape}\n")
    print(f"X_test_tensor shape: {X_test_tensor.shape}\n")
    print(f"y_test_tensor shape: {y_test_tensor.shape}\n")



def get_iris_dataloaders(batch_size=16, full_batch=False):
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = get_iris_data_tensors()

    # Determine batch sizes based on the flag
    if full_batch:
        train_batch_size = X_train_tensor.size(0)
        test_batch_size = X_test_tensor.size(0)
    else:
        train_batch_size = batch_size
        test_batch_size = batch_size
        
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


def plot_iris_losses(loss_lists, labels, filename="iris_loss_plot.png"):
    """
    Plots multiple lists of loss values on a single figure, regardless of length.

    Args:
        loss_lists (list of lists/arrays): The loss values for each training run.
        labels (list of str): Names for the legend of each run.
    """
    plt.figure(figsize=(10, 6))

    for i, loss_values in enumerate(loss_lists):
        # The X-axis for each list is simply its index, starting from 1 (iteration/epoch number)
        iterations = range(1, len(loss_values) + 1)

        # Plot the loss data
        plt.plot(iterations, loss_values, label=labels[i], alpha=0.8)

    # --- 3. Formatting the Plot ---
    plt.title('Comparison of Training Loss Histories')
    plt.xlabel('Iteration / Epoch Number')
    plt.ylabel('Loss Value')
    plt.legend(title='Type of Training')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout() # Adjusts plot to prevent labels from overlapping
    #plt.show()
    plt.savefig(filename)
    print(f"Training plot saved in file: {filename}")



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
