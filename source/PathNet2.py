import torch
import heapq
import random
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import time
import json
import pandas as pd

class QuantizedMLP:
    """ 
    A class representing a quantized MLP model.
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
        """Quantizes and clips a single parameter tensor in-place."""
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
    """
    def __init__(self, quantized_mlp, g_val, h_val, parent=None):         
        self.quantized_mlp = quantized_mlp
        self.g_val = g_val
        self.h_val = h_val
        self.f_val = self.g_val + self.h_val
        self.parent = parent

    def __lt__(self, other):
        return self.f_val < other.f_val
    

def get_neighbors(search_node, X, Y, quantization_factor=None, weight_kernel=[2,2], bias_kernel=[2], x_stride=1, y_stride=1, delta_abs=None):

    if quantization_factor is None:
        quantization_factor = search_node.quantized_mlp.quantization_factor

    neighbors = []
    parent_mlp = search_node.quantized_mlp
    parent_model = parent_mlp.model
    parent_parameter_list = list(parent_model.parameters())

    with torch.no_grad():
        #let's iterate over all parameters
        for tensor_index in range(len(parent_parameter_list)):
            #check if delta_abs is provided, otherwise use default 1/quantization_factor
            if delta_abs is None:
                delta_abs = 1 / quantization_factor
            # check if delta_abs is compatible with quantization factor (delta_abs must be always a multiple of 1/quantization_factor)
            else:
                if (delta_abs * quantization_factor) % 1 != 0:
                    raise ValueError(f"delta_abs={delta_abs} is not compatible with quantization_factor={quantization_factor}. The product delta_abs * quantization_factor must be an integer.")
                
            for delta in [+delta_abs, -delta_abs]:
                parent_tensor = list(parent_model.parameters())[tensor_index].data

                # check if any overflow would occur
                """
                if (torch.any(parent_tensor == parent_mlp.parameter_range[0]) and delta < 0) or \
                     (torch.any(parent_tensor == parent_mlp.parameter_range[1]) and delta > 0):
                    continue
                """
                
                # should be the right way to check for overflow, but need to be tested 
                if (torch.any(parent_tensor < parent_mlp.parameter_range[0] + delta_abs) and delta < 0) or \
                     (torch.any(parent_tensor > parent_mlp.parameter_range[1] - delta_abs) and delta > 0):
                    continue
                
                #check if tensor is 2D (weights)
                if len(parent_tensor.shape) == 2:
                    # check if the tensor is compatible with the weight kernel (has at least the size of the kernel)
                    if parent_tensor.shape[0] >= weight_kernel[0] and parent_tensor.shape[1] >= weight_kernel[1]:
                        # sliding window over the tensor
                        for i in range(0, parent_tensor.shape[0] - weight_kernel[0] + 1, y_stride):
                            for j in range(0, parent_tensor.shape[1] - weight_kernel[1] + 1, x_stride):
                                neighbor_mlp = deepcopy(parent_mlp)
                                neighbor_model = neighbor_mlp.model
                                target_tensor = list(neighbor_model.parameters())[tensor_index].data
                                window = target_tensor[i:i+weight_kernel[0], j:j+weight_kernel[1]]
                                #adding a small delta to each element in the window 
                                window += delta

                                # If a different quantization factor is provided, update the attribute and apply quantization to the entire model
                                if quantization_factor is not None:
                                    neighbor_mlp.quantization_factor = quantization_factor
                                    neighbor_mlp.quantize()
                                else:
                                    # Otherwise, only quantize the modified tensor
                                    # (this may be redundant if the summed delta is equal to 1/quantization_factor, but mandatory if delta assumes other values)
                                    neighbor_mlp.quantize_tensor(tensor_index)
                                
                                if neighbor_mlp.overflow:
                                    continue
                                
                                loss = neighbor_mlp.evaluate(X, Y)
                                neighbors.append((neighbor_mlp, loss))


                    # else the kernel is larger than the tensor itself and we take the whole tensor
                    else:
                        neighbor_mlp = deepcopy(parent_mlp)
                        neighbor_model = neighbor_mlp.model
                        target_tensor = list(neighbor_model.parameters())[tensor_index].data
                        #adding a small delta to each element in the tensor 
                        target_tensor += delta

                        # If a different quantization factor is provided, update the attribute and apply quantization to the entire model
                        if quantization_factor is not None:
                            neighbor_mlp.quantization_factor = quantization_factor
                            neighbor_mlp.quantize()
                        else:
                            # Otherwise, only quantize the modified tensor 
                            # (this may be redundant if the summed delta is equal to 1/quantization_factor, but mandatory if delta assumes other values)
                            neighbor_mlp.quantize_tensor(tensor_index)
                        
                        if neighbor_mlp.overflow:
                            continue
                        
                        loss = neighbor_mlp.evaluate(X, Y)
                        neighbors.append((neighbor_mlp, loss))

                # check if tensor is 1D (biases)
                elif len(parent_tensor.shape) == 1:
                    # check if the tensor is compatible with the bias kernel (has at least the size of the kernel)
                    if parent_tensor.shape[0] >= bias_kernel[0]:
                        # sliding window over the tensor
                        for i in range(0, parent_tensor.shape[0] - bias_kernel[0] + 1, y_stride):
                            neighbor_mlp = deepcopy(parent_mlp)
                            neighbor_model = neighbor_mlp.model
                            target_tensor = list(neighbor_model.parameters())[tensor_index].data
                            window = target_tensor[i:i+bias_kernel[0]]
                            #adding a small delta to each element in the window for demonstration
                            window += delta

                            # If a different quantization factor is provided, update the attribute and apply quantization to the entire model
                            if quantization_factor is not None:
                                neighbor_mlp.quantization_factor = quantization_factor
                                neighbor_mlp.quantize()
                            else:
                                # Otherwise, only quantize the modified tensor
                                # (this may be redundant if the summed delta is equal to 1/quantization_factor, but mandatory if delta assumes other values)
                                neighbor_mlp.quantize_tensor(tensor_index)
                            
                            if neighbor_mlp.overflow:
                                continue
                            
                            loss = neighbor_mlp.evaluate(X, Y)
                            neighbors.append((neighbor_mlp, loss))

                    # else the kernel is larger than the tensor itself and we take the whole tensor
                    else:
                        neighbor_mlp = deepcopy(parent_mlp)
                        neighbor_model = neighbor_mlp.model
                        target_tensor = list(neighbor_model.parameters())[tensor_index].data
                        #adding a small delta to each element in the tensor for demonstration
                        target_tensor += delta

                        # If a different quantization factor is provided, update the attribute and apply quantization to the entire model
                        if quantization_factor is not None:
                            neighbor_mlp.quantization_factor = quantization_factor
                            neighbor_mlp.quantize()
                        else:
                            # Otherwise, only quantize the modified tensor
                            # (this may be redundant if the summed delta is equal to 1/quantization_factor, but mandatory if delta assumes other values)
                            neighbor_mlp.quantize_tensor(tensor_index)
                        
                        if neighbor_mlp.overflow:
                            continue
                        
                        loss = neighbor_mlp.evaluate(X, Y)
                        neighbors.append((neighbor_mlp, loss))

                else:
                    #raise an error in the neighborhood generation process if tensor is neither 1D nor 2D
                    raise ValueError(f"Unsupported tensor shape at index {tensor_index}: {parent_tensor.shape}. Only 1D and 2D tensors are supported.")
                
    return neighbors



class Trainer:
    """
    A class to train a quantized MLP model using an A* search algorithm.
    """
    def __init__(self, model, loss_fn, quantization_factor, parameter_range, debug_mlp=True, 
                 #----------------------------------------------------------------------------------
                 weight_kernel = [2,2], bias_kernel = [2], x_stride=1, y_stride=1, delta_abs=None, 
                 #----------------------------------------------------------------------------------
                 early_stopping=False, e_s_patience=250,
                 #----------------------------------------------------------------------------------
                 dynamic_quantization=False, d_q_patience=100, 
                 quantization_factor_multiplier=10, max_quantization_factor=1e4,
                 #-----------------------------------------------------------------------------------
                 dynamic_kernel_reshaping=False, d_k_r_patience=100, 
                 x_weight_kernel_decr=1, y_weight_kernel_decr=1, y_bias_kernel_decr=1, 
                 min_weight_kernel=[1,1], min_bias_kernel=[1],
                 x_stride_decr=0, y_stride_decr=0, min_x_stride=1, min_y_stride=1,
                 #----------------------------------------------------------------------------------
                 loss_improvement_threshold=1e-5,
                 #----------------------------------------------------------------------------------
                 max_iterations=1000, log_freq=1000, measure_time=True, save_trained_model=False, model_name='best_model'):
        
        
        self.model = model          # nn.sequential model
        self.loss_fn = loss_fn
        self.quantization_factor = quantization_factor
        self.parameter_range = parameter_range
        self.debug_mlp = debug_mlp

        # Neighborhood Generation Parameters
        self.weight_kernel = weight_kernel
        self.bias_kernel = bias_kernel
        self.x_stride = x_stride
        self.y_stride = y_stride
        self.delta_abs = delta_abs

        # Early Stopping Parameters
        self.early_stopping = early_stopping
        self.e_s_patience = e_s_patience
        self.e_s_wait = 0           # counter for early stopping patience

        # Dynamic Quantization Parameters
        self.dynamic_quantization = dynamic_quantization
        self.d_q_patience = d_q_patience
        self.quantization_factor_multiplier = quantization_factor_multiplier    
        self.max_quantization_factor = max_quantization_factor
        self.d_q_wait = 0           # counter for dynamic quantization patience

        # Dynamic Kernel Reshaping Parameters
        self.dynamic_kernel_reshaping = dynamic_kernel_reshaping
        self.d_k_r_patience = d_k_r_patience
        self.x_weight_kernel_decr = x_weight_kernel_decr
        self.y_weight_kernel_decr = y_weight_kernel_decr
        self.y_bias_kernel_decr = y_bias_kernel_decr
        self.min_weight_kernel = min_weight_kernel
        self.min_bias_kernel = min_bias_kernel
        self.x_stride_decr = x_stride_decr
        self.y_stride_decr = y_stride_decr
        self.min_x_stride = min_x_stride
        self.min_y_stride = min_y_stride
        self.d_k_r_wait = 0         # counter for dynamic kernel reshaping patience

        # Loss Improvement Threshold for Early Stopping and Dynamic Adjustments (kernel reshaping, quantization)
        self.loss_improvement_threshold = loss_improvement_threshold

        self.max_iterations = max_iterations
        self.log_freq = log_freq

        self.open_set = []
        self.g_costs = {}       # It represents the best g-cost found so far for each MLP state
        self.best_node = None

        self.loss_history = []
        self.f_history = []
        self.g_history = []

        self.measure_time = measure_time
        self.save_trained_model = save_trained_model
        self.model_name = model_name
        self.training_time = None

    
    def dynamic_reshape_kernels_and_strides(self):
        """
        Dynamically reshapes the weight and bias kernels as well as the strides based on the defined decrements and minimum sizes.
        """
        # Reshape weight kernel
        if self.weight_kernel[0] > self.min_weight_kernel[0]:
            self.weight_kernel[0] = max(self.weight_kernel[0] - self.y_weight_kernel_decr, self.min_weight_kernel[0])
        if self.weight_kernel[1] > self.min_weight_kernel[1]:
            self.weight_kernel[1] = max(self.weight_kernel[1] - self.x_weight_kernel_decr, self.min_weight_kernel[1])
        
        # Reshape bias kernel
        if self.bias_kernel[0] > self.min_bias_kernel[0]:
            self.bias_kernel[0] = max(self.bias_kernel[0] - self.y_bias_kernel_decr, self.min_bias_kernel[0])
        
        # Adjust strides
        if self.x_stride > self.min_x_stride:
            self.x_stride = max(self.x_stride - self.x_stride_decr, self.min_x_stride)
        if self.y_stride > self.min_y_stride:
            self.y_stride = max(self.y_stride - self.y_stride_decr, self.min_y_stride)

    
    def reset_dynamic_counters(self):
        """
        Resets the patience counters for early stopping, dynamic quantization, and dynamic kernel reshaping.
        """
        if self.early_stopping: self.e_s_wait = 0
        if self.dynamic_quantization: self.d_q_wait = 0
        if self.dynamic_kernel_reshaping: self.d_k_r_wait = 0


    def increment_dynamic_counters(self):        
        """
        Increments the patience counters for early stopping, dynamic quantization, and dynamic kernel reshaping.
        The method also checks if the patience thresholds are reached to trigger dynamic adjustments.
        """
        if self.early_stopping: self.e_s_wait += 1

        if self.dynamic_quantization: 
            self.d_q_wait += 1
            # Check if it's time to adjust quantization factor
            if self.d_q_wait >= self.d_q_patience:
                new_qf = min(self.quantization_factor * self.quantization_factor_multiplier, self.max_quantization_factor)
                if new_qf > self.quantization_factor:
                    print(f"Dynamic Quantization applied: prev_qf={self.quantization_factor}, new_qf={new_qf}")
                    self.quantization_factor = new_qf
                self.d_q_wait = 0  # reset the counter after adjustment

        if self.dynamic_kernel_reshaping: 
            self.d_k_r_wait += 1
            # Check if it's time to reshape kernels and strides
            if self.d_k_r_wait >= self.d_k_r_patience:
                self.dynamic_reshape_kernels_and_strides()
                print(f"Dynamic Kernel Reshaping applied:\n prev_weight_kernel={self.weight_kernel}, prev_bias_kernel={self.bias_kernel}, prev_x_stride={self.x_stride}, prev_y_stride={self.y_stride}")
                print(f"new_weight_kernel={self.weight_kernel}, new_bias_kernel={self.bias_kernel}, new_x_stride={self.x_stride}, new_y_stride={self.y_stride}")
                self.d_k_r_wait = 0  # reset the counter after reshaping


    def train(self, X, Y):
        """
        Trains the quantized MLP using a corrected A* search algorithm.

        Parameters:
            X (torch.Tensor): Input data for training.
            Y (torch.Tensor): Target labels for training.
        """
        start_time = 0
        if self.measure_time:
            start_time = time.perf_counter()

        initial_mlp = QuantizedMLP(self.model, self.loss_fn, self.quantization_factor, self.parameter_range, debug=self.debug_mlp)
        initial_loss = initial_mlp.evaluate(X, Y)

        initial_node = SearchNode(quantized_mlp=initial_mlp, g_val=0, h_val=initial_loss)
        initial_hash = initial_mlp.get_state_hash()

        heapq.heappush(self.open_set, (initial_node.f_val, initial_node))
        self.g_costs[initial_hash] = initial_node.g_val
        self.best_node = initial_node

        for iteration in range(self.max_iterations):
            if not self.open_set:
                print("Open set is empty. Terminating search.")
                break

            current_f, current_node = heapq.heappop(self.open_set)

            current_hash = current_node.quantized_mlp.get_state_hash()
            # CHECK FOR STALE NODES
            # (current_hash not in self.g_costs) should always be FALSE, kept for security in short-circuit logic for 
            # the possible key error of the second expression
            if current_hash not in self.g_costs or current_node.g_val > self.g_costs[current_hash]:
                continue

            self.loss_history.append(current_node.h_val)
            self.f_history.append(current_node.f_val)
            self.g_history.append(current_node.g_val)


            #==================================================================================================
            # EARLY STOPPING AND DYNAMIC ADJUSTMENTS CHECK
            if self.best_node.h_val - current_node.h_val > self.loss_improvement_threshold:
                self.reset_dynamic_counters()
            else:
                self.increment_dynamic_counters()

            if self.early_stopping and self.e_s_wait >= self.e_s_patience:
                print(f"Early stopping triggered after {self.e_s_patience} iterations without improvement.")
                break
            #==================================================================================================


            if current_node.h_val < self.best_node.h_val:
                self.best_node = current_node
                print(f"Iteration {iteration+1}: New best loss = {self.best_node.h_val}")

            if (iteration + 1) % self.log_freq == 0:
                print(f"Iteration {iteration+1}: Best current loss = {self.best_node.h_val}")
                
            neighbors = get_neighbors(current_node, X, Y, self.quantization_factor, self.weight_kernel, self.bias_kernel, self.x_stride, self.y_stride, self.delta_abs)

            for neighbor_mlp, neighbor_loss in neighbors:
                if neighbor_mlp.overflow: continue
                neighbor_state_hash = neighbor_mlp.get_state_hash()

                g_step = neighbor_loss - current_node.h_val     # c(n,n') = loss(n') - loss(n) -> difference between child and parent loss

                # New g-cost (cost-to-come) for this neighbor
                g = current_node.g_val + g_step
                
                # Check if the neighbor state has not been visited yet or if this path offers a better g-cost (reinsertion case of the same MLP state to the open set)
                if neighbor_state_hash not in self.g_costs or g < self.g_costs[neighbor_state_hash]:  
                    self.g_costs[neighbor_state_hash] = g

                    # Create and push the new search node onto the open set, could be a real new state or an improved path to an existing state
                    new_node = SearchNode(neighbor_mlp, g_val=g, h_val=neighbor_loss, parent=current_node)
                    heapq.heappush(self.open_set, (new_node.f_val, new_node))

        print(f"Search completed after {iteration+1} iterations.")
        print(f"Best loss found: {self.best_node.h_val}")
        if self.measure_time:
            end_time = time.perf_counter()
            total_time = end_time - start_time
            self.training_time = total_time
            print(f"Total training time: {total_time:.4f} seconds")

        if self.save_trained_model:
            self.save_model(filename=self.model_name + '.pth')
        return

    def plot_training_history(self, filename='astar_loss_plot.png'):
        """
        Plots the loss (h), the total cost (f) and the cost g per iteration over all the iterations and saves the plot to a file.

        Parameters:
            filename (str): The name of the file to save the plot.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, label='Loss (h) per Iteration')
        plt.plot(self.f_history, label='Total Cost (f) per Iteration')
        plt.plot(self.g_history, label='Cost g per Iteration')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Value')
        plt.title('Loss (h), Total Cost (f) and Cost (g) Over Iterations with A*')
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

    def log_to_txt_file(self, filename='training_log.txt'):
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
            f.write(f"Training Time (seconds): {self.training_time}\n")
        print(f"Training log saved to {filename}")

    def log_to_json_file(self, filename='training_log.json'):
        """
        Logs the training history to a specified JSON file.

        Parameters:
            filename (str): The name of the JSON file to log the training history.
        """
        log_data = {
            "loss_history": self.loss_history,
            "f_history": self.f_history,
            "g_history": self.g_history,
            "best_loss": self.best_node.h_val,
            "training_time_seconds": self.training_time
        }
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=4)
        print(f"Training log saved to {filename}")

                

class GridSearchTrainer:
    """
    A class to perform grid search over multiple hyperparameter combinations for training quantized MLPs.
    """
    def __init__(self, models, loss_funcs, quantization_factors, parameter_ranges, weight_kernels, bias_kernels, x_strides, y_strides, max_iterations, log_freq, delta_abs=[None], debug_mlps=True, measure_time=True):
       
        self.trainers_params = []
        self.grid_search_data = []

        for i in range(len(models)):
            for lf in loss_funcs:
                for qf in quantization_factors:
                    for pr in parameter_ranges:
                        for wk in weight_kernels:
                            for bk in bias_kernels:
                                for xst in x_strides:
                                    for yst in y_strides:
                                        for da in delta_abs:
                                            for mi in max_iterations:
                                                for lfq in log_freq:
                                                    self.trainers_params.append((
                                                        models[i],
                                                        lf,
                                                        qf,
                                                        pr,
                                                        debug_mlps,
                                                        wk,
                                                        bk,
                                                        xst,
                                                        yst,
                                                        da,
                                                        mi,
                                                        lfq,
                                                        measure_time
                                                    ))


    def run_grid_search(self, X, Y, runs_per_config=1, enable_training_history_logging=True, log_filename='grid_search_results', save_models=False):
        """
        Runs the grid search over all trainer configurations and logs the results to a JSON file.

        Parameters:
            X (torch.Tensor): Input data for training.
            Y (torch.Tensor): Target labels for training.
            log_filename (str): Filename for the JSON log file.
        
        Returns:
            list: The list of results dictionaries.
        """

        with open(log_filename + '.txt', 'w') as log_file:
            log_file.write("=" * 32 + "\n")
            log_file.write("\tGrid Search Training Log\n")
            log_file.write("=" * 32 + "\n\n")
        
        print("\n" + "=" * 50)
        print("\tGrid Search")
        print("=" * 50)
        
        for config_index, param_config in enumerate(self.trainers_params):
            (model_class, loss_fn, qf, pr, debug_mlp, wk, bk, xst, yst, da, mi, lfq, measure_time) = param_config
            
            hyperparams_dict = {
                "model_type": str(model_class),
                "loss_fn": str(loss_fn),
                "quantization_factor": qf,
                "parameter_range": pr,
                "weight_kernel": wk,
                "bias_kernel": bk,
                "x_stride": xst,
                "y_stride": yst,
                "delta_abs": da,
                "max_iterations": mi,
                "log_freq": lfq,
                "debug_mlp": debug_mlp,
                "measure_time": measure_time
            }

            for run in range(runs_per_config):
                # Copying the model architecture but reinitializing weights for each run
                model = deepcopy(model_class)
                # random initialization of model weights
                for layer in model.modules():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()        

                trainer = Trainer(
                    model=model,
                    loss_fn=loss_fn,
                    quantization_factor=qf,
                    parameter_range=pr,
                    debug_mlp=debug_mlp,
                    weight_kernel=wk,
                    bias_kernel=bk,
                    x_stride=xst,
                    y_stride=yst,
                    delta_abs=da,
                    max_iterations=mi,
                    log_freq=lfq,
                    measure_time=measure_time
                )
                
                run_label = f"Config {config_index+1}/{len(self.trainers_params)} (Run {run+1}/{runs_per_config})"
                print(f"\n--- {run_label} ---")
                print(f"HPs: QF={qf}, PR={pr}, WK={wk}, BK={bk}, XS={xst},YS={yst}, DA={da}, MI={mi}, LFQ={lfq}\n")

                with open(log_filename + '.txt', 'a') as log_file:
                    log_file.write(f"(Run {run}) - Training with parameters: Quantization Factor={trainer.quantization_factor}, Parameter Range={trainer.parameter_range}, Weight Kernel={trainer.weight_kernel}, Bias Kernel={trainer.bias_kernel}, X Stride={trainer.x_stride}, Y Stride={trainer.y_stride}, Delta Abs={trainer.delta_abs}, Max Iterations={trainer.max_iterations}, Log Freq={trainer.log_freq}\n\n")

                trainer.train(X, Y)

                final_loss = trainer.best_node.h_val
                training_time = trainer.training_time if trainer.training_time else 0.0

                run_result = {
                    "config_index": config_index,
                    "run_number": run,
                    "hyperparameters": hyperparams_dict,
                    "metrics": {
                        "final_loss": final_loss,
                        "training_time_seconds": training_time
                    }
                }
                
                if enable_training_history_logging:
                    run_result["loss_history"] = trainer.loss_history
                    run_result["f_history"] = trainer.f_history
                    run_result["g_history"] = trainer.g_history

                self.grid_search_data.append(run_result)
                
                print(f"COMPLETED: Best Loss: {final_loss:.6f} | Time: {training_time:.2f}s")

                with open(log_filename + '.txt', 'a') as log_file:
                    log_file.write(f"(Run {run}) - Best Loss: {trainer.best_node.h_val}\n")
                    log_file.write(f"(Run {run}) - Training Time: {trainer.training_time:.4f} seconds\n")
                    log_file.write(f"\n(Run {run}) - Training completed.\n\n")
                    log_file.write("-" * 150 + "\n\n")

                if save_models:
                    model_filename = f"model_config{config_index}_run{run}.pth"
                    trainer.save_model(filename=model_filename)
        
        print("\n" + "-" * 50)
        print(f"Grid Search completed. Writing {len(self.grid_search_data)} results in {log_filename + '.json'}...")
        
        with open(log_filename + '.json', 'w') as f:
            json.dump(self.grid_search_data, f, indent=4)
            
        print("JSON write completed.")
        
        sorted_results = sorted(self.grid_search_data, key=lambda x: x["metrics"]["final_loss"])
        with open(log_filename + '.txt', 'a') as log_file:
            log_file.write("Sorted Final Losses from Grid Search:\n")
            for i, res in enumerate(sorted_results):
                hps = res['hyperparameters']
                log_file.write(f"{i+1}. Loss: {res['metrics']['final_loss']:.6f}\n")
                log_file.write(f"Model: {hps['model_type']}\n")
                log_file.write(f"QF: {hps['quantization_factor']}, PR: {hps['parameter_range']}, WK: {hps['weight_kernel']}, BK: {hps['bias_kernel']}, XS: {hps['x_stride']}, YS: {hps['y_stride']}, DA: {hps['delta_abs']}, MI: {hps['max_iterations']}\n\n")
        
    
    

    def plot_grid_search_trend(self, log_filename='grid_search_results', metric='loss_history'):
        """
        Reads the JSON log file and plots the loss trend (loss vs. iteration) for each run.
        
        Parameters:
            log_filename (str): The filename prefix for the JSON results file.
            metric (str): The key in the run_result dictionary containing the list of losses 
                          (e.g., 'loss_history', 'f_history' or 'g_history').
        """

        json_file = log_filename if log_filename.endswith('.json') else log_filename + '.json'

        try:
            with open(json_file, 'r') as f:
                results = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found at {json_file}. Please run grid search logging first.")
            return
        except json.JSONDecodeError:
            print(f"Error: Unable to decode JSON file {json_file}. File may be corrupted.")
            return
        
        if not results:
            print("No results found in the JSON file.")
            return

        plt.figure(figsize=(14, 8))
        
        print(f"\nGenerating trend plot for history key '{metric}'...")
        
        plotted_runs = 0
        for run_result in results:
            history_list = run_result.get(metric) 
            
            if history_list and isinstance(history_list, list) and history_list:
                try:
                    history_df = pd.DataFrame({
                        'iteration': range(1, len(history_list) + 1),
                        'history_value': history_list          
                    })
                    
                    # Labels for the plot legend
                    hps = run_result["hyperparameters"]
                    label = (
                        f"Config Index {run_result['config_index']} - Run N° {run_result['run_number']} "
                        f"[PR: {hps['parameter_range']}, QF: {hps['quantization_factor']}, WK: {hps['weight_kernel']}, BK: {hps['bias_kernel']}, XS: {hps['x_stride']}, YS: {hps['y_stride']}, DA: {hps['delta_abs']}, MI: {hps['max_iterations']}]"
                    )
                    
                    # X = iteration, Y = history_value
                    plt.plot(history_df['iteration'], history_df['history_value'], label=label, alpha=0.8, linewidth=1.5)
                    plotted_runs += 1

                except Exception as e:
                    print(f"Warning: Error plotting run {run_result['config_index']}-{run_result['run_number']}: {e}")
                    continue

        if plotted_runs == 0:
            print(f"No useful history data found in the JSON file for key '{metric}'.")
            plt.close() 
            return
        
        metric2name = {
            'loss_history': 'Loss',
            'f_history': 'Total Cost (f)',
            'g_history': 'Cost (g)'
        }
        metric_fullname = metric2name.get(metric)
        
        # Plot customization
        plt.title(f'{metric_fullname} Trend Across Grid Search Runs', fontsize=16)
        plt.xlabel('Iteration (Number of Steps)', fontsize=14)
        plt.ylabel(f'{metric_fullname} Value', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Legend outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title="Hyperparameter Configurations") 
        plt.tight_layout(rect=[0, 0, 1.00, 1]) # needed to avoid cutting off the legend
        plt.savefig(log_filename + '_loss_trend.png', dpi=300)

        print("Plot saved as " + log_filename + "_loss_trend.png\n")
        print("-" * 50)


    # Method to plot the average loss with 1 std shading for each configuration across runs, using numpy for mean and std calculations
    def plot_avg_loss(self, file_name='grid_search_avg_loss', parameter_name=None):
        """
        Plots the average loss with standard deviation shading for each hyperparameter configuration across multiple runs.

        Parameters:
            file_name (str): The filename prefix for saving the plot.
            parameter_name (str): The name of the hyperparameter being varied (for labeling purposes).
        """

        json_file = file_name if file_name.endswith('.json') else file_name + '.json'

        try:
            with open(json_file, 'r') as f:
                results = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found at {json_file}. Please run grid search logging first.")
            return
        except json.JSONDecodeError:
            print(f"Error: Unable to decode JSON file {json_file}. File may be corrupted.")
            return

        if not results:
            print("No results found in the JSON file.")
            return

        # Organize losses by configuration index
        config_losses = {}
        config_labels = {}
        for run_result in results:
            config_index = run_result['config_index']
            loss_history = run_result.get('loss_history', [])
            if config_index not in config_labels:
                # checking if the delta_abs parameter is None to print it properly in the label
                if run_result["hyperparameters"]["delta_abs"] is None:
                    run_result["hyperparameters"]["delta_abs"] = f"1/{run_result['hyperparameters']['quantization_factor']}"
                hps = run_result["hyperparameters"]
                if parameter_name is None:
                    config_labels[config_index] = (
                        f"Config {config_index} [PR: {hps['parameter_range']}, QF: {hps['quantization_factor']}, "
                        f"WK: {hps['weight_kernel']}, BK: {hps['bias_kernel']}, XS: {hps['x_stride']}, YS: {hps['y_stride']},"
                        f"DA: {hps['delta_abs']}, MI: {hps['max_iterations']}]"
                )
                else:
                    #check if parameter_name is a list or a single string
                    if isinstance(parameter_name, list):
                        labels = []
                        for pn in parameter_name:
                            param_value = run_result["hyperparameters"].get(pn)
                            labels.append(f"{pn}: {param_value}")
                        config_labels[config_index] = ", ".join(labels)
                    else:
                        # if a specific parameter_name is provided, only show that parameter in the label
                        param_value = run_result["hyperparameters"].get(parameter_name)
                        config_labels[config_index] = f"{parameter_name}: {param_value}"
            if loss_history:
                if config_index not in config_losses:
                    config_losses[config_index] = []
                config_losses[config_index].append(loss_history)

        plt.figure(figsize=(14, 8))

        for config_index, loss_lists in config_losses.items():
            # Convert to numpy array for easier mean/std calculation
            loss_array = np.array(loss_lists)
            
            # Calculate mean and std deviation across runs
            mean_loss = np.mean(loss_array, axis=0)
            std_loss = np.std(loss_array, axis=0)

            iterations = range(1, len(mean_loss) + 1)

            # Plot mean loss
            plt.plot(iterations, mean_loss, label=config_labels[config_index], linewidth=2)

            # Plot std deviation shading
            plt.fill_between(iterations, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)

        # Plot customization
        plt.title('Average Loss with Standard Deviation Across Grid Search Configurations', fontsize=16)
        plt.xlabel('Iteration (Number of Steps)', fontsize=14)
        plt.ylabel('Loss Value', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)

        # Legend outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title="Hyperparameters Tested") 
        plt.tight_layout(rect=[0, 0, 1.00, 1]) # needed to avoid cutting off the legend
        plt.savefig(file_name + '_avg_loss.png', dpi=300)
        print("Plot saved as " + file_name + "_avg_loss.png\n")


    # Method to plot boxplot of final losses for each configuration across runs 
    def plot_final_loss_boxplot(self, file_name='grid_search_final_loss_boxplot', x_label=None):
        """
        Plots a boxplot of the final losses for each hyperparameter configuration across multiple runs.

        Parameters:
            file_name (str): The filename prefix for saving the plot.
            y_label (str): The label for the y-axis. If None, defaults to 'Final Loss'.
        """

        json_file = file_name if file_name.endswith('.json') else file_name + '.json'

        try:
            with open(json_file, 'r') as f:
                results = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found at {json_file}. Please run grid search logging first.")
            return
        except json.JSONDecodeError:
            print(f"Error: Unable to decode JSON file {json_file}. File may be corrupted.")
            return

        if not results:
            print("No results found in the JSON file.")
            return

        # Organize final losses by configuration index
        config_final_losses = {}
        config_labels = {}
        for run_result in results:
            config_index = run_result['config_index']
            final_loss = run_result['metrics'].get('final_loss')
            if config_index not in config_labels:
                # checking if the delta_abs parameter is None to print it properly in the label
                if run_result["hyperparameters"]["delta_abs"] is None:
                    run_result["hyperparameters"]["delta_abs"] = f"1/{run_result['hyperparameters']['quantization_factor']}"
                if x_label is None:
                    hps = run_result["hyperparameters"]
                    config_labels[config_index] = (
                        f"Config {config_index} [PR: {hps['parameter_range']}, QF: {hps['quantization_factor']}, "
                        f"WK: {hps['weight_kernel']}, BK: {hps['bias_kernel']}, XS: {hps['x_stride']}, YS: {hps['y_stride']},"
                        f"DA: {hps['delta_abs']}, MI: {hps['max_iterations']}]"
                    )
                else:
                    #checking if the x_labels is a list or a single string
                    if isinstance(x_label, list):
                        labels = []
                        for xl in x_label:
                            param_value = run_result["hyperparameters"].get(xl)
                            labels.append(f"{xl}: {param_value}")
                        config_labels[config_index] = ", ".join(labels)
                    else:
                        # if a specific x_label is provided, only show that parameter in the label
                        param_value = run_result["hyperparameters"].get(x_label)
                        config_labels[config_index] = f"{x_label}: {param_value}"
            if final_loss is not None:
                if config_index not in config_final_losses:
                    config_final_losses[config_index] = []
                config_final_losses[config_index].append(final_loss)

        # Prepare data for boxplot
        boxplot_data = []
        boxplot_labels = []
        for config_index, losses in config_final_losses.items():
            boxplot_data.append(losses)
            boxplot_labels.append(config_labels[config_index])

        plt.figure(figsize=(14, 8))
        #plt.boxplot(boxplot_data, labels=boxplot_labels, showfliers=True)
            # Boxplot showing median, IQR, and range
        plt.boxplot(boxplot_data, vert=True, patch_artist=True, labels=boxplot_labels, 
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='darkred'))
        #plt.xticks(rotation=45, ha='right')

        # adding jittered scatter points for each final loss
        for i, losses in enumerate(boxplot_data):
            y = losses
            x = np.random.normal(i + 1, 0.04, size=len(y))  # Adding some jitter to the x-axis
            plt.scatter(x, y, alpha=0.6, color='blue', s=20)

        plt.title('Final Loss Distribution Across Grid Search Configurations', fontsize=16)
        plt.xlabel("Hyperparameter Tested", fontsize=14)
        plt.ylabel('Final Loss', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(file_name + '_boxplot.png', dpi=300)
        print("Boxplot saved as " + file_name + "_boxplot.png\n")


    # Method to generate a statistical summary of final losses for each configuration across runs and save it to a text file
    def generate_final_loss_summary(self, file_name='grid_search_final_loss_summary'):
        """
        Generates a summary table of final losses for each hyperparameter configuration across multiple runs and saves it to a text file.

        Parameters:
            file_name (str): The filename prefix for saving the summary text file.
        """

        json_file = file_name if file_name.endswith('.json') else file_name + '.json'

        try:
            with open(json_file, 'r') as f:
                results = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found at {json_file}. Please run grid search logging first.")
            return
        except json.JSONDecodeError:
            print(f"Error: Unable to decode JSON file {json_file}. File may be corrupted.")
            return

        if not results:
            print("No results found in the JSON file.")
            return

        # Organize final losses and training times by configuration index
        config_final_losses = {}
        config_training_times = {}
        config_labels = {}
        for run_result in results:
            config_index = run_result['config_index']
            final_loss = run_result['metrics'].get('final_loss')
            training_time = run_result['metrics'].get('training_time_seconds', 0.0)
            if config_index not in config_labels:
                hps = run_result["hyperparameters"]
                if run_result["hyperparameters"]["delta_abs"] is None:
                    run_result["hyperparameters"]["delta_abs"] = f"1/{run_result['hyperparameters']['quantization_factor']}"
                config_labels[config_index] = (
                    f"Config {config_index} [PR: {hps['parameter_range']}, QF: {hps['quantization_factor']}, "
                    f"WK: {hps['weight_kernel']}, BK: {hps['bias_kernel']}, XS: {hps['x_stride']}, YS: {hps['y_stride']},"
                    f"DA: {hps['delta_abs']}, MI: {hps['max_iterations']}]"
                )
            if final_loss is not None:
                if config_index not in config_final_losses:
                    config_final_losses[config_index] = []
                    config_training_times[config_index] = []
                config_final_losses[config_index].append(final_loss)
                config_training_times[config_index].append(training_time)

        # Write summary to text file
        summary_file = file_name + '_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("=" * 120 + "\n")
            f.write("Final Loss Summary Across Grid Search Configurations\n")
            f.write("=" * 120 + "\n\n\n\n")
            # writing a table mapping each config index to its hyperparameter settings
            f.write("=" * 120 + "\n")
            f.write("Configuration Index to Hyperparameter Settings Mapping:\n")
            f.write("=" * 120 + "\n\n")
            for config_index, label in config_labels.items():
                f.write(f"{config_index}: {label}\n")
                f.write("-" * 120 + "\n")

            f.write("=" * 120 + "\n\n\n\n")

            f.write("=" * 120 + "\n\n")
            f.write(f"{'Config':<10}{'Mean Loss':<15}{'Median Loss':<15}{'Std Dev':<15}{'Variance':<15}{'Min Loss':<15}{'Max Loss':<15}{'Mean Time (s)':<15}\n")
            f.write("-" * 120 + "\n")
            for config_index, losses in config_final_losses.items():
                training_times = config_training_times[config_index]
                mean_loss = np.mean(losses)
                median_loss = np.median(losses)
                std_loss = np.std(losses)
                variance_loss = np.var(losses)
                min_loss = np.min(losses)
                max_loss = np.max(losses)
                mean_time = np.mean(training_times) if training_times else 0.0
                f.write(f"{config_index:<10}{mean_loss:<15.6f}{median_loss:<15.6f}{std_loss:<15.6f}{variance_loss:<15.6f}{min_loss:<15.6f}{max_loss:<15.6f}{mean_time:<15.2f}\n")
            f.write("-" * 120 + "\n")