#===================================================================================================================================
#===================================================================================================================================
#-------------- run this file from project root: python -m astar_optimization.simple_dataset_test.circle_test ----------------------
#===================================================================================================================================
#===================================================================================================================================

import torch
import torch.nn as nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import numpy as np
from source.circle_utils import plot_circles_dataset
from source.PathNet import Trainer
from source.general_utils import plot_losses


# Simple neural network model for circle classification
model = nn.Sequential(
    nn.Linear(2, 4),  
    nn.ReLU(),
    nn.Linear(4, 2)   
)

# Parameters for synthetic dataset
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

#plot_circles_dataset(X, y, noise_level=noise_level, factor_level=factor_level, n_samples=n_samples)

# Stratify over y (labels) to maintain class proportions in train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_seed, stratify=y
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

print("\n--- Data Preparation Summary ---")
print(f"Input Feature Count (input_size): {X.shape[1]}")        # Should be 2
print(f"Output Class Count (num_classes): {len(np.unique(y))}") # Should be 2
print(f"Training Set Size: {X_train_tensor.shape[0]}")
print(f"Test Set Size: {X_test_tensor.shape[0]}\n")

trainer = Trainer(model, nn.CrossEntropyLoss(), quantization_factor=2, parameter_range=(-4, 4), debug_mlp=True, \
                  weight_kernel=[2,2], bias_kernel=[2], stride=1, delta_abs=None, max_iterations=1000, log_freq=100, \
                    measure_time=True, save_trained_model=False, model_name="circle_classification_model")

trainer.train(X_train_tensor, y_train_tensor)

plot_losses([trainer.loss_history], ["A-Star"])

"""predictions = trainer.best_node.quantized_mlp.model(X_test_tensor)
correct = 0
for i, prediction in enumerate(predictions):
    predicted_class = torch.argmax(prediction).item()
    print(f"Input: {X_test_tensor[i].numpy()}, Predicted Class: {predicted_class}, Actual: {y_test_tensor[i].item()}")
    print(f"Raw Output: {prediction.detach().numpy()}\n")
    if predicted_class == y_test_tensor[i].item():
        correct += 1

accuracy = correct / len(y_test_tensor)
print(f"\n\nTest Accuracy: {accuracy * 100:.2f}%")"""