#===================================================================================================================================
#===================================================================================================================================
#--------------------- run this file from project root: python -m gradient_optimization.iris_grad_base -----------------------------
#===================================================================================================================================
#===================================================================================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

# --- 1. Data Preparation ---
iris = load_iris()
X, y = iris.data, iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert NumPy arrays to PyTorch Tensors
# Features should be float32, labels should be long (for CrossEntropyLoss)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for batching and shuffling
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


# --- 2. Model Definition (MLP) ---
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

# --- 3. Model Initialization ---
INPUT_SIZE = X.shape[1]  # 4 features
HIDDEN_SIZE = 10         # A simple choice for a small dataset
NUM_CLASSES = len(np.unique(y)) # 3 classes

model = IrisMLP(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)

# Define Loss function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- 4. Training Loop ---
NUM_EPOCHS = 100
loss_history = []

print("Starting Training...")
model.train() # Set the model to training mode

initial_time = time.perf_counter()

for epoch in range(NUM_EPOCHS):
    for X_batch, y_batch in train_loader:
        # 1. Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # 2. Backward pass and optimization
        optimizer.zero_grad() # Clear gradients from previous step
        loss.backward()       # Compute gradient of the loss w.r.t model parameters
        optimizer.step()      # Update the parameters

    loss_history.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

print("Training Complete.")

print(f"Total Training Time: {time.perf_counter() - initial_time:.2f} seconds\n\n")

# --- 5. Evaluation ---
model.eval() # Set the model to evaluation mode (disables dropout, etc.)
with torch.no_grad(): # Disable gradient calculation for efficiency
    test_outputs = model(X_test_tensor)
    # Get the predicted class by finding the index of the max log-probability
    _, predicted = torch.max(test_outputs.data, 1)
    
    # Calculate Accuracy
    total = y_test_tensor.size(0)
    correct = (predicted == y_test_tensor).sum().item()
    accuracy = 100 * correct / total

print(f'\nTest Accuracy: {accuracy:.2f}%')