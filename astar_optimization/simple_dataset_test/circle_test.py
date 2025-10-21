#===================================================================================================================================
#===================================================================================================================================
#-------------- run this file from project root: python -m astar_optimization.simple_dataset_test.circle_test ----------------------
#===================================================================================================================================
#===================================================================================================================================

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from source.PathNet import Trainer

# --- 1. Generate the Circles Dataset ---

n_samples = 1000
noise_level = 0.1
factor_level = 0.5 
random_seed = 42

# X will have 2 features, y will have 2 classes (0 or 1)
# class 0: inner circle, class 1: outer circle
X, y = make_circles(
    n_samples=n_samples,
    noise=noise_level, 
    factor=factor_level,
    random_state=random_seed
)

# -----------------------------------------------------
## 2. Plotting Logic
# -----------------------------------------------------

plt.figure(figsize=(6, 6))
# Create a scatter plot where the color 'c' is determined by the label 'y'.
# The 'coolwarm' colormap is useful for binary classification.
plt.scatter(
    X[:, 0], # Feature 1 (X-axis)
    X[:, 1], # Feature 2 (Y-axis)
    c=y, 
    cmap=plt.cm.coolwarm,
    edgecolor='k', # Black border around points
    s=40 # Size of points
)
plt.title(f"Synthetic Circles Dataset (Noise={noise_level}, Factor={factor_level})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True, linestyle='--', alpha=0.6)
#plt.show() # Display the plot
plt.savefig("circles_dataset.png") # Save the plot as a PNG file
# -----------------------------------------------------

# --- 3. Split Data ---
# Since it's a binary problem (0 and 1), we use stratify=y.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_seed, stratify=y
)

# --- 4. Convert to PyTorch Tensors ---
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# --- 5. Create PyTorch DataLoaders ---
batch_size = 32 
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- Information for your custom QuantizedMLP ---
print("\n--- Data Preparation Summary ---")
print(f"Input Feature Count (input_size): {X.shape[1]}") # Should be 2
print(f"Output Class Count (num_classes): {len(np.unique(y))}") # Should be 2
print(f"Training Set Size: {len(train_dataset)}")


model = nn.Sequential(
    nn.Linear(2, 4),  
    nn.ReLU(),
    nn.Linear(4, 2)   
)
trainer = Trainer(model, nn.CrossEntropyLoss(), quantization_factor=2, parameter_range=(-4, 4), debug_mlp=True, param_fraction=1.0, max_iterations=200, log_freq=500, target_loss=0.0001, update_strategy=2, g_ini_val=0, g_step=0.01, alpha=0.5, scale_f=True)

trainer.train(X_train_tensor, y_train_tensor)

predictions = trainer.best_node.quantized_mlp.model(X_test_tensor)
correct = 0
for i, prediction in enumerate(predictions):
    # Calculate the predicted class index (argmax over dimension 0 for a single prediction)
    predicted_class = torch.argmax(prediction).item()
    
    print(f"Input: {X_test_tensor[i].numpy()}, Predicted Class: {predicted_class}, Actual: {y_test_tensor[i].item()}")
    print(f"Raw Output: {prediction.detach().numpy()}\n")
    if predicted_class == y_test_tensor[i].item():
        correct += 1

accuracy = correct / len(y_test_tensor)
print(f"\n\nTest Accuracy: {accuracy * 100:.2f}%")