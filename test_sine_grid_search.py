#===================================================================================================================================
#===================================================================================================================================
#---------------------------------- run this file from project root: python -m test_sine_grid_search -------------------------------
#===================================================================================================================================
#===================================================================================================================================


from source.sinusoidal_func_utils import generate_sinusoidal_tensor
from source.PathNet2 import Trainer

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


NUM_SAMPLES = 1000
MIN_ANGLE = 0
MAX_ANGLE = 4 * np.pi
NOISE_LEVEL = 0.1
ITERATIONS = 5000

RUNS = 2

LOG_FILE_ASTAR = "sine_model_astar_multiple_runs"

model = nn.Sequential(
        nn.Linear(1, 4),  
        nn.ReLU(),
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
        nn.Tanh()
        )

X_train, Y_train = generate_sinusoidal_tensor(func=torch.sin, num_samples=NUM_SAMPLES, min_angle=MIN_ANGLE, max_angle=MAX_ANGLE, noise_level=NOISE_LEVEL)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                                       INDIVIDUAL PARAMETER TESTS
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# testing kernel sizes and strides

config_results = []

weight_kernels_to_test = [[2,2], [3,3]]
bias_kernels_to_test = [[2], [3]]
strides = [1]  

for w_kernel in weight_kernels_to_test:
    for b_kernel in bias_kernels_to_test:
        for stride in strides:

            temp_dict = {
                "description": f"Weight Kernel: {w_kernel}, Bias Kernel: {b_kernel}, Stride: {stride}",
                "losses": [],
                "training_times": [],
                "final_losses": []
            }

            for run in range(RUNS):
                print(f"ASTAR Run {run+1}/{RUNS} with weight kernel {w_kernel}, bias kernel {b_kernel}, stride {stride}")
                astar_trainer = Trainer(model, nn.MSELoss(), quantization_factor=10, parameter_range=(-10, 10), debug_mlp=True, weight_kernel = w_kernel, bias_kernel = b_kernel, stride=stride, delta_abs=None, max_iterations=ITERATIONS, log_freq=1000)

                astar_trainer.train(X_train, Y_train)

                temp_dict["losses"].append(astar_trainer.training_history['loss_history'])
                temp_dict["training_times"].append(astar_trainer.training_time[-1])
                temp_dict["final_losses"].append(astar_trainer.best_node.h_val)
                
            config_results.append(temp_dict)


# need to be tested
def plot_mean_loss_per_configuration(config_results, iterations, filename="mean_loss_per_configuration.png"):
    """Plots the mean loss per configuration over iterations."""
    plt.figure(figsize=(10, 6))

    for config in config_results:
        losses_array = np.array(config["losses"])
        mean_losses = np.mean(losses_array, axis=0)
        plt.plot(range(1, iterations + 1), mean_losses, label=config["description"])

    plt.title('Mean Training Loss per Configuration')
    plt.xlabel('Iterations')
    plt.ylabel('Mean MSE Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(filename)
    print(f"Saved plot: {filename}")


# statistical summary ...