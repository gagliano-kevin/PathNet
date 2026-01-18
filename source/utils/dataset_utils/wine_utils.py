import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np



def prepare_data_tensors(X, y):
    # Split into Train and "Remainder" (Test + Val)
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8, random_state=42)
    
    # Split Remainder into Validation and Test (50/50 of the remaining 20%)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)
    
    # 3. Scaling
    scaler = StandardScaler()
    # ONLY fit on the training data to avoid data leakage
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # 4. Convert to Tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    
    # Helper to clean target data
    def to_long_tensor(target):
        if hasattr(target, 'values'):
            target_np = target.astype(int).values
        else:
            target_np = np.array(target).astype(int)
        return torch.tensor(target_np, dtype=torch.long)

    y_train = to_long_tensor(y_train)
    y_val = to_long_tensor(y_val)
    y_test = to_long_tensor(y_test)

    # 5. Label Shifting (Start at 0)
    min_label = y_train.min()
    if min_label > 0:
        y_train = y_train - min_label
        y_val = y_val - min_label
        y_test = y_test - min_label
        
    return X_train, y_train, X_val, y_val, X_test, y_test



def get_wine_data():
    data = fetch_openml(name="wine-quality-red", version=1, as_frame=True)
    X, y = data.data, data.target
    return prepare_data_tensors(X, y)



def create_dataloader(X, y, batch_size=None):
    if batch_size is None:
        batch_size = len(X)     # Full-batch training
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

