import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt



def get_splitted_iris_data_tensors():
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



def get_iris_data_tensors():
    iris = load_iris()
    X_train, y_train = iris.data, iris.target

    # Scaling features for better performance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
 
    return X_train_tensor, y_train_tensor



def print_iris_data_info():
    tensors = get_splitted_iris_data_tensors()
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = tensors

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
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = get_splitted_iris_data_tensors()

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



def get_train_iris_dataloader(batch_size=16, full_batch=False):
    X_train_tensor, y_train_tensor = get_iris_data_tensors()

    if full_batch:
        train_batch_size = X_train_tensor.size(0)
    else:
        train_batch_size = batch_size
        
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    return train_loader