import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def prepare_data_tensors(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    
    # Fix the "str" error: Convert y to numpy integers first
    # This handles Pandas Series, Categorical data, and String lists
    if hasattr(y_train, 'values'):
        y_train_np = y_train.astype(int).values
        y_test_np = y_test.astype(int).values
    else:
        y_train_np = np.array(y_train).astype(int)
        y_test_np = np.array(y_test).astype(int)

    y_train = torch.tensor(y_train_np, dtype=torch.long)
    y_test = torch.tensor(y_test_np, dtype=torch.long)

    # Shift labels to start at 0
    min_label = y_train.min()
    if min_label > 0:
        y_train = y_train - min_label
        y_test = y_test - min_label
        
    return X_train, y_train, X_test, y_test


def get_wine_data():
    data = fetch_openml(name="wine-quality-red", version=1, as_frame=True)
    X, y = data.data, data.target
    return prepare_data_tensors(X, y)


def create_dataloader(X, y, batch_size=None):
    if batch_size is None:
        batch_size = len(X)     # Full-batch training
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

