import torch
import heapq
import time
from source.utils.memory_guard import SystemMemoryGuard
from source.utils.neighbors_utils import get_neighbors


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
        """
        Quantizes and clips a single parameter tensor in-place.
        """
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
                 x_stride_decr=1, y_stride_decr=1, min_x_stride=1, min_y_stride=1,
                 #----------------------------------------------------------------------------------
                 loss_improvement_threshold=1e-5,
                 #----------------------------------------------------------------------------------
                 max_iterations=1000, log_freq=1000, measure_time=True, save_trained_model=False, model_name='best_model'):
        
        # Memory guard for stopping gracefully training when 90% of system memory usage is reached
        self.memory_guard = SystemMemoryGuard()

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

        # Container for storing iteration number in which dynamic adjustments were made
        self.dynamic_adjustments_log = {
            "dynamic_quantization_iterations": [],
            "dynamic_kernel_reshaping_iterations": []
        }


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


    def increment_dynamic_counters(self, iteration):        
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
                    self.dynamic_adjustments_log["dynamic_quantization_iterations"].append(iteration)
                    print(f"Dynamic Quantization applied: prev_qf={self.quantization_factor}, new_qf={new_qf}")
                    self.quantization_factor = new_qf
                self.d_q_wait = 0  # reset the counter after adjustment

        if self.dynamic_kernel_reshaping: 
            self.d_k_r_wait += 1
            # Check if it's time to reshape kernels and strides
            if self.d_k_r_wait >= self.d_k_r_patience:
                self.dynamic_reshape_kernels_and_strides()
                self.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"].append(iteration)
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
                self.increment_dynamic_counters(iteration)

            if self.early_stopping and self.e_s_wait >= self.e_s_patience:
                print(f"Early stopping triggered after {self.e_s_patience} iterations without improvement.")
                break

            if self.memory_guard.memory_exceeded():
                print("Memory usage exceeded threshold. Terminating training to prevent system instability.")
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


    def save_model(self, filename='best_model.pth'):
        """
        Saves the best model's state dictionary to a specified file.
        """
        if self.best_node is not None:
            torch.save(self.best_node.quantized_mlp.model.state_dict(), filename)
            print(f"Best model saved to {filename}")
        else:
            print("No best model to save.")


    def load_model(self, model_architecture, loss_fn, quantization_factor=10, parameter_range=(-5, 5), enable_quantization=True, debug=False, filename='best_model.pth'):
        """
        Loads a model's state dictionary from a specified file and returns a QuantizedMLP instance
        """
        state_dict = torch.load(filename, weights_only=True)
        model_architecture.load_state_dict(state_dict)
        quantized_mlp = QuantizedMLP(model_architecture, loss_fn, quantization_factor, parameter_range, enable_quantization, debug)
        print(f"Model loaded from {filename}")
        return quantized_mlp

