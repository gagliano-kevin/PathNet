#===================================================================================================================================
#===================================================================================================================================
#--------------------- run this file from project root: python -m gradient_optimization.iris_grad_base -----------------------------
#===================================================================================================================================
#===================================================================================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from source.iris_utils import get_iris_data_tensors, print_iris_data_info, get_iris_dataloaders, IrisMLP

print_iris_data_info()

X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = get_iris_data_tensors()

# Full batch mode enabled in order to be consistent with astar's full batch training 
# (number of epochs in gradient base training is coherent to the number of iterations of astar search)
train_loader, test_loader = get_iris_dataloaders(batch_size=16, full_batch=True)

# Model parameters
INPUT_SIZE = 4          # 4 features
HIDDEN_SIZE = 10        
NUM_CLASSES = 3         # 3 classes

model = IrisMLP(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)

# Loss function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
NUM_EPOCHS = 100
loss_history = []

print("Starting Training...")
model.train() # Set the model to training mode

initial_time = time.perf_counter()

for epoch in range(NUM_EPOCHS):
    for X_batch, y_batch in train_loader:
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass and optimization
        optimizer.zero_grad() # Clear gradients from previous step
        loss.backward()       # Compute gradient of the loss w.r.t model parameters
        optimizer.step()      # Update the parameters

    loss_history.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

print("Training Complete.")

print(f"Total Training Time: {time.perf_counter() - initial_time:.2f} seconds\n\n")

# Evaluation
model.eval()                # Set the model to evaluation mode (disables dropout, etc.)
with torch.no_grad():       # Disable gradient calculation for efficiency
    test_outputs = model(X_test_tensor)
    # Get the predicted class by finding the index of the max log-probability
    _, predicted = torch.max(test_outputs.data, 1)
    
    # Calculate Accuracy
    total = y_test_tensor.size(0)
    correct = (predicted == y_test_tensor).sum().item()
    accuracy = 100 * correct / total

print(f'\nTest Accuracy: {accuracy:.2f}%')
