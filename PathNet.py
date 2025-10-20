import torch
import torch.nn as nn
import heapq
import random
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

"""
    Simple MLP model for XOR problem
"""
XOR_MODEL = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)


"""
    XOR dataset and labels
"""
XOR_DATASET = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
XOR_LABELS = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


"""    
    Loss function for XOR problem
"""
XOR_LOSS_FN = nn.BCELoss()
        


class QuantizedMLP:
    """ 
    A class representing a quantized MLP model for the XOR problem.
    It includes methods for quantization, evaluation, and state management.
    """
    def __init__(self, model, loss_fn, quantization_factor=10, parameter_range=(-5, 5), enable_quantization=True, debug=False):
        self.model = model
        self.loss_fn = loss_fn
        self.quantization_factor = quantization_factor
        self.parameter_range = parameter_range
        self.overflow = False
        self.enable_quantization = enable_quantization
        self.debug = debug
        self.possible_congigurations = ((2 * parameter_range[1] * quantization_factor) + 1) ** len(self.get_flat_weights())
        if self.enable_quantization: self.quantize()

    def quantize(self):
        """
        Quantizes the model parameters to a discrete set of floating-point values.
        The step size is 1/quantization_factor, and values are clipped to the parameter_range.
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # Check for overflow before quantization
                if torch.any(param.data < self.parameter_range[0]) or torch.any(param.data > self.parameter_range[1]):
                    if self.debug: print(f"Debug Warning in quantize(): Parameter '{name}' with the value {param.data} is outside the specified range. Clipping to range.")
                    param.data.clamp_(self.parameter_range[0], self.parameter_range[1])

                # Quantize by rounding to the nearest multiple of 1/quantization_factor
                param.data.mul_(self.quantization_factor).round_().div_(self.quantization_factor)

    def quantize_tensor(self, tensor_idx):
        """Quantizes and clips a single parameter in-place."""
        with torch.no_grad():
            tensor_to_quantize = list(self.model.parameters())[tensor_idx]
            if torch.any(tensor_to_quantize.data < self.parameter_range[0]) or torch.any(tensor_to_quantize.data > self.parameter_range[1]):
                if self.debug: print(f"Debug Warning: Parameter with the value {tensor_to_quantize.data} outside the specified range. Setting model to None and rising overflow flag.")
                self.overflow = True
                self.model = None
                return
            tensor_to_quantize.data.mul_(self.quantization_factor).round_().div_(self.quantization_factor)

    def evaluate(self, X, Y):
        """
        Evaluates the model on the given data.
        """
        if self.model is None or self.overflow:
            raise ValueError("Model is not valid due to overflow or quantization issues.")
        self.model.eval()
        with torch.no_grad():
            return self.loss_fn(self.model(X), Y).item()

    def get_flat_weights(self):
        """
        Returns a flattened tensor of all model parameters.
        """
        if self.model is None or self.overflow:
            raise ValueError("Model is not valid due to overflow or quantization issues.")
        return torch.cat([p.detach().flatten() for p in self.model.parameters()])
    
    def get_state_hash(self):
        """
        Generates a hashable tuple representing the state of the model's quantized weights.
        """
        if self.model is None or self.overflow:
            raise ValueError("Model is not valid due to overflow or quantization issues.")
        # The product by quantization factor ensures the hash is based on integer representations of the quantized values 
        return tuple((self.get_flat_weights() * self.quantization_factor).long().tolist())
    
    def __str__(self):
        repr = f"QuantizedMLP(quantization_factor={self.quantization_factor}, parameter_range={self.parameter_range}, overflow={self.overflow})"
        repr += "\nModel Parameters:\n"
        for name, param in self.model.named_parameters():
            repr += f"{name}: {param.data}\n"
        return repr
    


class SearchNode:
    """ 
    A class representing a node in the search space for the quantized MLP.
    It contains the quantized MLP, its evaluation scores, and a reference to its parent node.
    Currently g_fn, h_fn and f_fn are not used, but they can be implemented for custom cost functions.
    """
    def __init__(self, quantized_mlp, g_val, h_val, f_val=None, g_fn=None, h_fn=None, f_fn=None, parent=None):         
        self.quantized_mlp = quantized_mlp
        self.g_val = g_val
        self.h_val = h_val
        self.f_val = self.g_val + self.h_val if f_val is None else f_val
        self.g_fn = g_fn                    # not used currently
        self.h_fn = h_fn                    # not used currently
        self.f_fn = f_fn                    # not used currently
        self.parent = parent

    def __lt__(self, other):
        return self.f_val < other.f_val
    


def get_neighbors(search_node, X, Y, quantization_factor=None, param_fraction=1):
    """
    Generates neighbors for the given search node by modifying a fraction (param_fraction) 
    of scalar weights within every parameter tensor.
    
    Parameters:
        search_node (SearchNode): The current search node containing the quantized MLP.
        X (torch.Tensor): Input data for evaluation.
        Y (torch.Tensor): Target labels for evaluation.
        quantization_factor (int, optional): The quantization factor to use. If None, uses the parent's factor.
        param_fraction (float): Fraction (p) of SCALAR WEIGHTS to modify within *each* parameter tensor.

    Returns:
        list: A list of tuples containing the neighbor MLP and its evaluation score.
    """
    if quantization_factor is None:
        quantization_factor = search_node.quantized_mlp.quantization_factor

    neighbors = []
    parent_mlp = search_node.quantized_mlp
    parent_model = parent_mlp.model
    
    # Get the list of parameter tensors from the parent model
    parent_parameters = list(parent_model.parameters())
    
    with torch.no_grad():
        # Iterate over each parameter tensor in the model
        for tensor_idx in range(len(parent_parameters)):
            
            # Clone the original tensor data
            original_data = parent_parameters[tensor_idx].data.clone()
            
            # Get the total number of scalar elements in the selected tensor
            num_elements = original_data.numel()
            
            num_scalars_to_change = max(1, int(num_elements * param_fraction))
            selected_scalar_indexes = random.sample(range(num_elements), num_scalars_to_change)
            
            # Iterate inside the selected tensor (layer parameters, weights or biases) in order to create neighbors by modifying selected scalars
            for flat_scalar_idx in selected_scalar_indexes:
                
                # Get the scalar value at the chosen index
                scalar_value = original_data.view(-1)[flat_scalar_idx].item()
                
                for delta in [-(1 / quantization_factor), (1 / quantization_factor)]:
                    
                    # Boundary check - skip if the change would go out of bounds
                    if (scalar_value == parent_mlp.parameter_range[0] and delta < 0) or \
                       (scalar_value == parent_mlp.parameter_range[1] and delta > 0):
                        continue

                    # Create a deep copy of the parent MLP to modify
                    neighbor_mlp = deepcopy(parent_mlp)
                    
                    # Get the parameter tensor to modify
                    tensor_to_modify = list(neighbor_mlp.model.parameters())[tensor_idx]
                    
                    # Clone the tensor data to modify
                    new_tensor = tensor_to_modify.data.clone() 
                    
                    # Apply the delta to the specific index of the cloned tensor
                    new_tensor.view(-1)[flat_scalar_idx] = scalar_value + delta
                    
                    # Update the parameter's data
                    tensor_to_modify.data = new_tensor
                    

                    # If a different quantization factor is provided, update the attribute and apply quantization to the entire model
                    if quantization_factor is not None:
                        neighbor_mlp.quantization_factor = quantization_factor
                        neighbor_mlp.quantize()
                    else:
                        # Otherwise, only quantize the modified tensor
                        neighbor_mlp.quantize_tensor(tensor_idx)
                    
                    if neighbor_mlp.overflow:
                        continue
                    
                    h = neighbor_mlp.evaluate(X, Y)
                    neighbors.append((neighbor_mlp, h))
                    
    return neighbors



class Trainer:
    """
    A class to train a quantized MLP model using an A* search algorithm.
    """
    def __init__(self, model, loss_fn, quantization_factor, parameter_range, debug_mlp=True, param_fraction=1, max_iterations=1000, log_freq=1000, target_loss=0.1, update_strategy=0, g_ini_val=0.001, g_step=0.01, alpha=0.5, scale_f=True):
        self.model = model
        self.loss_fn = loss_fn
        self.quantization_factor = quantization_factor
        self.parameter_range = parameter_range
        self.debug_mlp = debug_mlp
        self.param_fraction = param_fraction

        self.max_iterations = max_iterations
        self.log_freq = log_freq
        self.target_loss = target_loss
        self.update_strategy = update_strategy
        
        # Initial g value is zero for update strategies 1 and 2
        self.g_initial_value = g_ini_val if update_strategy not in [1,2] else 0

        # Parameter used only for update strategy 0
        self.g_step = g_step

        # Parameter used only for update strategy 1
        self.alpha = alpha

        # Parameter used only for update strategy 1 and 2
        self.scale_f = scale_f

        self.open_set = []
        self.visited = {}
        self.best_node = None

        self.loss_history = []
        self.f_history = []
        self.g_history = []

    def train(self, X, Y):
        """
        Trains the quantized MLP using the A* search algorithm.

        Parameters:
            X (torch.Tensor): Input data for training.
            Y (torch.Tensor): Target labels for training.
        """
        initial_mlp = QuantizedMLP(self.model, self.loss_fn, self.quantization_factor, self.parameter_range, debug=self.debug_mlp)
        initial_loss = initial_mlp.evaluate(X, Y)
        initial_node = SearchNode(quantized_mlp=initial_mlp, g_val=self.g_initial_value, h_val=initial_loss)

        heapq.heappush(self.open_set, (initial_node.f_val, initial_node))
        self.visited[initial_mlp.get_state_hash()] = initial_node.f_val
        self.best_node = initial_node

        for iteration in range(self.max_iterations):
            if not self.open_set:
                print("Open set is empty. Terminating search.")
                break

            current_f, current_node = heapq.heappop(self.open_set)
            
            self.loss_history.append(current_node.h_val)
            self.f_history.append(current_node.f_val)
            self.g_history.append(current_node.g_val)

            if current_node.h_val < self.best_node.h_val:
                self.best_node = current_node
                print(f"Iteration {iteration+1}: New best loss = {self.best_node.h_val}")

            if current_node.h_val <= self.target_loss:
                print(f"Goal loss achieved: {current_node.h_val}")
                print(f"Training completed in {iteration+1} iterations.")
                self.best_node = current_node
                return

            if iteration % self.log_freq == 0:
                print(f"Iteration {iteration+1}: Best current loss = {self.best_node.h_val}")

            neighbors = get_neighbors(current_node, X, Y, self.quantization_factor, self.param_fraction)

            for neighbor_mlp, h in neighbors:
                if neighbor_mlp.overflow: continue
                state_hash = neighbor_mlp.get_state_hash()
                if self.update_strategy == 0:
                    g = current_node.g_val + self.g_step
                    f = g + h
                elif self.update_strategy == 1:
                    g = current_node.g_val + (1/np.log(initial_node.quantized_mlp.possible_congigurations)) * (self.alpha * h/current_node.h_val + (1-self.alpha) * h/initial_node.h_val)
                    f = max(0, 1-(self.target_loss/h))*g + h if self.scale_f else g + h
                elif self.update_strategy == 2:
                    g = current_node.g_val + (1/self.max_iterations)
                    f = max(0, 1-(self.target_loss/h))*g + h if self.scale_f else g + h
                # Exception of type KeyError is prevented by prior checking if state_hash is not in visited (short-circuit logic)
                if state_hash not in self.visited or f < self.visited[state_hash]:
                    self.visited[state_hash] = f
                    new_node = SearchNode(neighbor_mlp, g_val=g, h_val=h, f_val=f, parent=current_node)
                    heapq.heappush(self.open_set, (new_node.f_val, new_node))

        print(f"Search completed after {iteration+1} iterations.")
        print(f"Best loss found: {self.best_node.h_val}")
        return 

    def plot(self, filename='astar_loss_plot.png'):
        """
        Plots the loss (h) and total cost (f) over iterations and saves the plot to a file.

        Parameters:
            filename (str): The name of the file to save the plot.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, label='Loss (h) per Iteration')
        plt.plot(self.f_history, label='Total Cost (f) per Iteration')
        plt.plot(self.g_history, label='Cost g per Iteration')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Value')
        plt.title('Loss (h) and Total Cost (f) Over Iterations with A*')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        print(f"Training plot saved in file: {filename}")

    def save_model(self, filename='best_model.pth'):
        """
        Saves the best model's state dictionary to a specified file.

        Parameters:
            filename (str): The name of the file to save the model.
        """
        if self.best_node is not None:
            torch.save(self.best_node.quantized_mlp.model.state_dict(), filename)
            print(f"Best model saved to {filename}")
        else:
            print("No best model to save.")

    def load_model(self, model_architecture, loss_fn, quantization_factor=10, parameter_range=(-5, 5), enable_quantization=True, debug=False, filename='best_model.pth'):
        """
        Loads a model's state dictionary from a specified file and returns a QuantizedMLP instance
        
        Parameters:
            model_architecture (nn.Module): The architecture of the model to load.
            loss_fn (callable): The loss function to use.
            quantization_factor (int): The quantization factor to use.
            parameter_range (tuple): The parameter range for quantization.
            enable_quantization (bool): Whether to enable quantization.
            debug (bool): Whether to enable debug mode.
            filename (str): The name of the file to load the model from.
            
            Returns:
            QuantizedMLP: The loaded quantized MLP model.
        """
        state_dict = torch.load(filename, weights_only=True)
        model_architecture.load_state_dict(state_dict)
        quantized_mlp = QuantizedMLP(model_architecture, loss_fn, quantization_factor, parameter_range, enable_quantization, debug)
        print(f"Model loaded from {filename}")
        return quantized_mlp


    def log_to_file(self, filename='training_log.txt'):
        """
        Logs the training history to a specified file.

        Parameters:
            filename (str): The name of the file to log the training history.
        """
        with open(filename, 'a') as f:
            f.write("Iteration\tLoss (h)\tTotal Cost (f)\tCost g\n")
            for i in range(len(self.loss_history)):
                f.write(f"{i+1}\t{self.loss_history[i]}\t{self.f_history[i]}\t{self.g_history[i]}\n")
            f.write(f"\nBest Loss: {self.best_node.h_val}\n\n")
        print(f"Training log saved to {filename}")



class GridSearchTrainer:
    """
    A class to perform grid search over multiple hyperparameter combinations for training quantized MLPs.
    """
    def __init__(self, models, loss_funcs, quantization_factors, parameter_ranges, param_fractions, max_iterations, log_freq, target_losses, update_strategies, g_ini_vals, g_steps, alphas, scale_fs, debug_mlps=True):
        self.trainers = []
        for i in range(len(models)):
            for lf in loss_funcs:
                for qf in quantization_factors:
                    for pr in parameter_ranges:
                        for pf in param_fractions:
                            for mi in max_iterations:
                                for lfq in log_freq:
                                    for tl in target_losses:
                                        for us in update_strategies:
                                            for giv in g_ini_vals:
                                                for gs in g_steps:
                                                    for a in alphas:
                                                        for sf in scale_fs:
                                                            trainer = Trainer(
                                                                model=models[i],
                                                                loss_fn=lf,
                                                                quantization_factor=qf,
                                                                parameter_range=pr,
                                                                debug_mlp=debug_mlps,
                                                                param_fraction=pf,
                                                                max_iterations=mi,
                                                                log_freq=lfq,
                                                                target_loss=tl,
                                                                update_strategy=us,
                                                                g_ini_val=giv,
                                                                g_step=gs,
                                                                alpha=a,
                                                                scale_f=sf
                                                            )
                                                            self.trainers.append(trainer)
    
    def run_grid_search(self, X, Y, log_filename='grid_search_log.txt'):
        """
        Runs the grid search over all trainer configurations and logs the results.

        Parameters:
            X (torch.Tensor): Input data for training.
            Y (torch.Tensor): Target labels for training.
            log_filename (str): Filename to log the training results.
        """

        sorted_final_losses = []

        with open(log_filename, 'w') as log_file:
            log_file.write("Grid Search Training Log\n")
            log_file.write("=" * 80 + "\n\n")

        for trainer in self.trainers:
            print(f"Starting training with parameters: Quantization Factor={trainer.quantization_factor}, Parameter Range={trainer.parameter_range}, Param Fraction={trainer.param_fraction}, Max Iterations={trainer.max_iterations}, Target Loss={trainer.target_loss}, Update Strategy={trainer.update_strategy}, G Initial Value={trainer.g_initial_value}, G Step={trainer.g_step}, Alpha={trainer.alpha}, Scale F={trainer.scale_f}")
            with open(log_filename, 'a') as log_file:
                log_file.write(f"Training with parameters: Quantization Factor={trainer.quantization_factor}, Parameter Range={trainer.parameter_range}, Param Fraction={trainer.param_fraction}, Max Iterations={trainer.max_iterations}, Target Loss={trainer.target_loss}, Update Strategy={trainer.update_strategy}, G Initial Value={trainer.g_initial_value}, G Step={trainer.g_step}, Alpha={trainer.alpha}, Scale F={trainer.scale_f}\n\n")
            trainer.train(X, Y)
            heapq.heappush(sorted_final_losses, (trainer.best_node.h_val, [trainer.quantization_factor, trainer.parameter_range, trainer.param_fraction, trainer.max_iterations, trainer.target_loss, trainer.update_strategy, trainer.g_initial_value, trainer.g_step, trainer.alpha, trainer.scale_f]))
            trainer.log_to_file(log_filename)
            print("Training completed.\n\n")
            with open(log_filename, 'a') as log_file:
                log_file.write("\n\n Training completed.\n\n")
                log_file.write("-" * 80 + "\n\n")

        with open(log_filename, 'a') as log_file:
            log_file.write("Sorted Final Losses from Grid Search:\n")
            log_file.write("Final Loss\t\t\t\tParameters: [Quantization Factor, Parameter Range, Param Fraction, Max Iterations, Target Loss, Update Strategy, G Initial Value, G Step, Alpha, Scale F]\n")
            while sorted_final_losses:
                loss, params = heapq.heappop(sorted_final_losses)
                log_file.write(f"{loss}\t\t\t{params}\n")




if __name__ == '__main__':

    model = nn.Sequential(
    nn.Linear(2, 2),
    nn.ReLU(),
    nn.Linear(2, 1),
    nn.Sigmoid()
    ) 
    """
    trainer = Trainer(model, nn.MSELoss(), quantization_factor=2, parameter_range=(-4, 4), debug_mlp=True, param_fraction=1.0, max_iterations=2000, log_freq=1000, target_loss=0.0001, update_strategy=2, g_ini_val=0, g_step=0.01, alpha=0.5, scale_f=True)

    trainer.train(XOR_DATASET, XOR_LABELS)

    trainer.plot('xor_base.png')

    trainer.log_to_file('xor_base_log.txt')

    predictions = trainer.best_node.quantized_mlp.model(XOR_DATASET)
    print("\n\nPredictions on XOR dataset:")
    print(predictions.detach().numpy())

    # Print best model parameters
    best_model = trainer.best_node.quantized_mlp.model
    print("\n\nBest model parameters:")
    for name, param in best_model.named_parameters():
        print(f"{name}: {param.data}")

    trainer.save_model('xor_best_model.pth')
    quantized_mlp = trainer.load_model(model_architecture=model, filename='xor_best_model.pth', loss_fn=nn.MSELoss(), quantization_factor=2, parameter_range=(-4, 4), enable_quantization=True, debug=True)
    quantized_mlp.evaluate(XOR_DATASET, XOR_LABELS)

    print("\n\nQuantized MLP after loading from file:")
    print(quantized_mlp)
    """
    grid_search_trainer = GridSearchTrainer(
        models=[model],
        loss_funcs=[nn.MSELoss()],
        quantization_factors=[1, 2],
        #quantization_factors=[1, 2, 5],
        parameter_ranges=[(-5, 5)],
        #parameter_ranges=[(-4, 4), (-5, 5)],
        param_fractions=[1.0],
        #param_fractions=[0.5, 1.0],
        max_iterations=[1000],
        log_freq=[500],
        target_losses=[0.0001],
        update_strategies=[2],
        #update_strategies=[0, 1, 2],
        g_ini_vals=[0],
        g_steps=[0.01],
        alphas=[0.5],
        scale_fs=[True],
        #scale_fs=[True, False],
        debug_mlps=True
    )

    grid_search_trainer.run_grid_search(XOR_DATASET, XOR_LABELS, log_filename='xor_grid_search_log.txt')