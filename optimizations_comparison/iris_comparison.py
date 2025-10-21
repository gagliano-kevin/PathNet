#===================================================================================================================================
#===================================================================================================================================
#-------------- run this file from project root: python -m optimizations_comparison.iris_comparison ------------------------
#===================================================================================================================================
#===================================================================================================================================

import torch
import torch.nn as nn
from source.PathNet import Trainer
from source.iris_utils import get_iris_data_tensors, print_iris_data_info, plot_iris_losses, get_iris_dataloaders, IrisMLP

import torch.optim as optim
import numpy as np

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- ASTAR TRAINING -----------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

print_iris_data_info()

X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = get_iris_data_tensors()

# Simple neural network model for iris classification
model = nn.Sequential(
nn.Linear(4, 4),
nn.ReLU(),
nn.Linear(4, 3),
) 

trainer = Trainer(model, nn.CrossEntropyLoss(), quantization_factor=2, parameter_range=(-4, 4), debug_mlp=True, param_fraction=1.0, max_iterations=200, log_freq=500, target_loss=0.0001, update_strategy=2, g_ini_val=0, g_step=0.01, alpha=0.5, scale_f=True)

trainer.train(X_train_tensor, y_train_tensor)


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------- GRADIENT BASE TRAINING -------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

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
NUM_EPOCHS = 200
loss_history = []

print("Starting Gradient Base Training...")
model.train() # Set the model to training mode

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


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------


losses = [trainer.loss_history, loss_history]
loss_labels = ["A-Star", "Gradient base"]

plot_iris_losses(losses, loss_labels)