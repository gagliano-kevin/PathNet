#===================================================================================================================================
#===================================================================================================================================
#-------------- run this file from project root: python -m astar_optimization.simple_dataset_test.sine_test ----------------------
#===================================================================================================================================
#===================================================================================================================================
from source.sinusoidal_func_utils import generate_sinusoidal_tensor, plot_sine_predictions
#from source.PathNet import Trainer
from source.SimplePathNet import Trainer

import torch
import torch.nn as nn
import numpy as np

NUM_SAMPLES = 10000
MIN_ANGLE = 0
MAX_ANGLE = 4 * np.pi
NOISE_LEVEL = 0.1

X_sin, Y_sin = generate_sinusoidal_tensor(func=torch.sin, num_samples=NUM_SAMPLES, min_angle=MIN_ANGLE, max_angle=MAX_ANGLE, noise_level=NOISE_LEVEL)

"""model = nn.Sequential(
    nn.Linear(1, 6),  
    nn.ReLU(),
    #nn.Tanh(),
    nn.Linear(6, 4),
    nn.ReLU(),
    #nn.Tanh(),
    nn.Linear(4, 1),
    #nn.Tanh()
)"""


model = nn.Sequential(
    nn.Linear(1, 8),
    nn.Tanh(), 

    nn.Linear(8, 8),
    nn.Tanh(),

    nn.Linear(8, 1),
    nn.Tanh() # Tanh bounds the output to (-1, 1), matching the sine range.
)

#trainer = Trainer(model, nn.MSELoss(), quantization_factor=5, parameter_range=(-4, 4), debug_mlp=True, param_fraction=1.0, max_iterations=1000, log_freq=50, target_loss=0.1, update_strategy=0, g_ini_val=0, g_step=None, alpha=0.5, scale_f=False)

trainer = Trainer(model, nn.MSELoss(), quantization_factor=10, parameter_range=(-4, 4), debug_mlp=True, param_fraction=1.0, max_iterations=1000, log_freq=100, target_loss=0.01)

trainer.train(X_sin, Y_sin)

plot_sine_predictions(test_x_np=X_sin.numpy(), 
                      predicted_sin_np=trainer.best_node.quantized_mlp.model(X_sin).detach().numpy(), 
                      true_sin_np=Y_sin.numpy(),
                      filename="sine_model_astar_test.png")

