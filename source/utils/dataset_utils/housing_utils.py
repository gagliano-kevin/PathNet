import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def prepare_data_tensors(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
 
    return X_train, y_train, X_test, y_test



def get_california_housing_data():
    data = fetch_california_housing()
    X, y = data.data, data.target
    return prepare_data_tensors(X, y)



def create_dataloader(X, y, batch_size=None):
    if batch_size is None:
        batch_size = len(X)     # Full-batch training
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

