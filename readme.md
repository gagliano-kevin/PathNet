PathNet: A* Search Trainer for Quantized MLPs 🧠🗺️

PathNet is a Python-based implementation that explores a novel approach to training a Multi-Layer Perceptron (MLP) neural network by framing the weight optimization problem as a shortest path search using the A* algorithm.

Instead of traditional gradient-based methods (like backpropagation), this project treats each unique combination of quantized network weights as a state in a search graph. The A* algorithm then efficiently searches this state space to find the weight configuration (the path) that minimizes the network's loss function.

Features ✨

A* Search Optimization: Uses the heuristic-driven A* algorithm to find optimal quantized weight configurations, where the loss function acts as the heuristic ($h$) to guide the search.

Quantized Weights: Implements the QuantizedMLP class to discretize network weights, creating a finite, searchable state space.

XOR Example: Includes a ready-to-run implementation for solving the classic XOR problem.

Flexible Trainer: The Trainer class allows for customization of quantization parameters, A* cost function variants (via update_strategy), and termination conditions.

Hyperparameter Grid Search: The GridSearchTrainer facilitates automated testing across various hyperparameter combinations.

Logging and Visualization: Includes methods to plot loss history and log training results to file.

Getting Started ⚙️

Prerequisites

You'll need Python 3 and the following libraries:

pip install torch numpy matplotlib


Quick Run (XOR Example)

The main script is configured to demonstrate the training process using the XOR dataset and a basic grid search.

Clone the repository or save the code as pathnet_trainer.py.

Run the script from your terminal:

python pathnet_trainer.py


The script will:

Perform a grid search over specified hyperparameters.

Log the training process and final results to xor_grid_search_log.txt.

Print the best loss found across all configurations.

Core Classes and Concepts 💡

QuantizedMLP

This class wraps a standard torch.nn.Module and enforces quantization on its weights.

Quantization: Weights are rounded to the nearest multiple of $\frac{1}{\text{quantization\_factor}}$. This converts the infinite, continuous space of weights into a finite, discrete, and searchable state space.

State Hashing: The get_state_hash() method converts the discrete weights into a hashable tuple, which is used by the A* algorithm to track visited states and prevent cycles.

SearchNode

Represents a single state in the search space.

$\mathbf{h}$ Value (Heuristic): The immediate loss of the quantized_mlp on the training data.

$\mathbf{g}$ Value (Path Cost): The accumulated cost of reaching the current state from the initial weight state.

$\mathbf{f}$ Value (Total Cost): $f = g + h$. This guides the priority queue during the search.

Trainer

Implements the A* search logic.

Neighbor Generation: The get_neighbors() function generates neighboring states by applying a small perturbation ($\pm \frac{1}{\text{quantization\_factor}}$) to a subset of the scalar weights. Each perturbation represents an action or step in the search path.

Update Strategies: The update_strategy parameter allows you to experiment with different $\mathbf{g}$ and $\mathbf{f}$ cost calculations, affecting the search path exploration (from fixed steps to adaptive, heuristic-scaled costs).