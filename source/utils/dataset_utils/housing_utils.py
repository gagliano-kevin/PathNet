import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



def prepare_data_tensors(X, y):
    # Split into Train and "Remainder" (Test + Val)
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.1, random_state=42)
    
    # Split Remainder into Validation and Test (50/50 of the remaining 20%)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)
    
    return (
        to_tensor(X_train), to_tensor(y_train).view(-1, 1),
        to_tensor(X_val), to_tensor(y_val).view(-1, 1),
        to_tensor(X_test), to_tensor(y_test).view(-1, 1)
    )



def get_california_housing_data():
    data = fetch_california_housing()
    X, y = data.data, data.target
    return prepare_data_tensors(X, y)



def create_dataloader(X, y, batch_size=None):
    if batch_size is None:
        batch_size = len(X)     # Full-batch training
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)



"""
    Function to evaluate regression model performance using common metrics.
    The method is tailored for standard pytorch models and dataloaders.
    For A-star PathNet models, a custom evaluation method may be needed.
"""
def evaluate_sgd_regression(model, dataloader):
    model.eval() # Set model to evaluation mode
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            preds = model(X_batch)
            all_preds.append(preds)
            all_targets.append(y_batch)
            
    # Concatenate and convert back to numpy for sklearn metrics
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()
    
    metrics = {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }
    
    return metrics


def evaluate_pathnet_regression(trainer, data):
    model = trainer.best_node.quantized_mlp.model
    model.eval()
    X, y = data
    with torch.no_grad():
        preds = model(X).numpy()
        y_true = y.numpy()
    
    metrics = {
        "MSE": mean_squared_error(y_true, preds),
        "RMSE": np.sqrt(mean_squared_error(y_true, preds)),
        "MAE": mean_absolute_error(y_true, preds),
        "R2": r2_score(y_true, preds)
    }

    return metrics
    