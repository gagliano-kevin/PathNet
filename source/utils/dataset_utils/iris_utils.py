import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np



def get_splitted_iris_data_tensors():
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split into Train and "Remainder" (Test + Val)
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8, random_state=42)
    
    # Split Remainder into Validation and Test (50/50 of the remaining 20%)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)

    # 3. Scaling: Fit ONLY on training data, transform others
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 4. Convert to Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor



def print_iris_data_info():
    tensors = get_splitted_iris_data_tensors()
    X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor = tensors

    print("--- Iris Data: Train, Val, and Test Splits ---")
    print(f"Input Feature Count: {X_train_tensor.shape[1]}")
    print(f"Output Class Count:  {len(np.unique(y_train_tensor.numpy()))}")
    print("-" * 30)
    print(f"Training Set Size:   {X_train_tensor.shape[0]}")
    print(f"Validation Set Size: {X_val_tensor.shape[0]}")
    print(f"Test Set Size:       {X_test_tensor.shape[0]}\n")



def get_iris_dataloaders(batch_size=None):
    tensors = get_splitted_iris_data_tensors()
    X_tr, y_tr, X_val, y_val, X_te, y_te = tensors

    # If no batch size is provided, use full dataset size
    if batch_size is None:
        tr_bs, val_bs, te_bs = X_tr.size(0), X_val.size(0), X_te.size(0)
    else:
        tr_bs = val_bs = te_bs = batch_size
        
    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=tr_bs, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=val_bs, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=te_bs, shuffle=False)

    return train_loader, val_loader, test_loader