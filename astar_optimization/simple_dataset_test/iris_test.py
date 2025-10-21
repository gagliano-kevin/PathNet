#===================================================================================================================================
#===================================================================================================================================
#-------------- run this file from project root: python -m astar_optimization.simple_dataset_test.iris_test ------------------------
#===================================================================================================================================
#===================================================================================================================================

import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from source.PathNet import Trainer

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

# Info of the dataset
print("--- Data Variables Ready for Custom Model ---")
print(f"Input Feature Count (input_size): {X_train.shape[1]}")
print(f"Output Class Count (num_classes): {len(np.unique(y))}")
print(f"Training Set Size: {X_train_tensor.shape[0]}")
print(f"Test Set Size: {len(X_test_tensor.shape[0])}\n")
print(f"X_train_tensor shape: {X_train_tensor.shape}\n")
print(f"y_train_tensor shape: {y_train_tensor.shape}\n")
print(f"X_test_tensor shape: {X_test_tensor.shape}\n")
print(f"y_test_tensor shape: {y_test_tensor.shape}\n")

# Simple neural network model for iris classification
model = nn.Sequential(
nn.Linear(4, 4),
nn.ReLU(),
nn.Linear(4, 3),
) 

trainer = Trainer(model, nn.CrossEntropyLoss(), quantization_factor=2, parameter_range=(-4, 4), debug_mlp=True, param_fraction=1.0, max_iterations=200, log_freq=500, target_loss=0.0001, update_strategy=2, g_ini_val=0, g_step=0.01, alpha=0.5, scale_f=True)

trainer.train(X_train_tensor, y_train_tensor)

predictions = trainer.best_node.quantized_mlp.model(X_test_tensor)
correct = 0
for i, prediction in enumerate(predictions):
    predicted_class = torch.argmax(prediction).item()
    print(f"Input: {X_test_tensor[i].numpy()}, Predicted Class: {predicted_class}, Actual: {y_test_tensor[i].item()}")
    print(f"Raw Output: {prediction.detach().numpy()}\n")
    if predicted_class == y_test_tensor[i].item():
        correct += 1

accuracy = correct / len(y_test_tensor)
print(f"\n\nTest Accuracy: {accuracy * 100:.2f}%")