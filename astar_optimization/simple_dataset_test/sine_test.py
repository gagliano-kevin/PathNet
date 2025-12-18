#===================================================================================================================================
#===================================================================================================================================
#-------------- run this file from project root: python -m astar_optimization.simple_dataset_test.sine_test ----------------------
#===================================================================================================================================
#===================================================================================================================================
from source.sinusoidal_func_utils import generate_sinusoidal_tensor, plot_sine_predictions, plot_sine_data
from source.PathNet import Trainer
import torch
import torch.nn as nn
import numpy as np

NUM_SAMPLES = 1000
MIN_ANGLE = 0
MAX_ANGLE = 4 * np.pi
NOISE_LEVEL = 0.1

X_sin, Y_sin = generate_sinusoidal_tensor(func=torch.sin, num_samples=NUM_SAMPLES, min_angle=MIN_ANGLE, max_angle=MAX_ANGLE, noise_level=NOISE_LEVEL)

model = nn.Sequential(
    nn.Linear(1, 4),  
    nn.ReLU(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Tanh()   # Tanh bounds the output to (-1, 1), matching the sine range
)
#PathNet version
#trainer = Trainer(model, nn.MSELoss(), quantization_factor=10, parameter_range=(-4, 4), debug_mlp=True, param_fraction=1.0, max_iterations=1000, log_freq=100, target_loss=0.01)

#PathNet2 version with sliding window kernels
trainer = Trainer(model, nn.MSELoss(), quantization_factor=10, parameter_range=(-4, 4), debug_mlp=True, weight_kernel = [3,3], bias_kernel = [3], stride=1, delta_abs=None, max_iterations=1000, log_freq=100)


trainer.train(X_sin, Y_sin)

trainer.plot_training_history("test.png")

plot_sine_predictions(test_x_np=X_sin.numpy(), 
                      predicted_sin_np=trainer.best_node.quantized_mlp.model(X_sin).detach().numpy(), 
                      true_sin_np=Y_sin.numpy(),
                      filename="sine_model_astar_test.png")

plot_sine_data(X_sin, Y_sin, filename="sine_dataset.png")