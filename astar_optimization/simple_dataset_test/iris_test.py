#===================================================================================================================================
#===================================================================================================================================
#-------------- run this file from project root: python -m astar_optimization.simple_dataset_test.iris_test ------------------------
#===================================================================================================================================
#===================================================================================================================================

import torch
import torch.nn as nn
from source.PathNet import Trainer
from source.iris_utils import get_iris_data_tensors, print_iris_data_info
from source.general_utils import plot_losses


print_iris_data_info()

X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = get_iris_data_tensors()

# Simple neural network model for iris classification
model = nn.Sequential(
nn.Linear(4, 4),
nn.ReLU(),
nn.Linear(4, 3),
) 

trainer = Trainer(model, nn.CrossEntropyLoss(), quantization_factor=2, parameter_range=(-4, 4), debug_mlp=True, \
                  weight_kernel=[2,2], bias_kernel=[2], stride=1, delta_abs=None, max_iterations=1000, log_freq=100, \
                    measure_time=True, save_trained_model=False, model_name="iris_classification_model")

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

losses = [trainer.loss_history]
loss_labels = ["A-Star"]

plot_losses(losses, loss_labels)