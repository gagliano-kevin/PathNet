import numpy as np
import torch
from copy import deepcopy



def get_neighbors_old_version(search_node, X, Y, quantization_factor=None, weight_kernel=[2,2], bias_kernel=[2], x_stride=1, y_stride=1, delta_abs=None):

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



"""
A revised version of the get_neighbors function that dynamically adjusts the kernel size and strides if the kernel is larger than the tensor itself.
"""
def get_neighbors(search_node, X, Y, quantization_factor=None, weight_kernel=[2,2], bias_kernel=[2], x_stride=1, y_stride=1, delta_abs=None):

    if quantization_factor is None:
        quantization_factor = search_node.quantized_mlp.quantization_factor

    neighbors = []
    parent_mlp = search_node.quantized_mlp
    parent_model = parent_mlp.model
    parent_parameter_list = list(parent_model.parameters())

    # Ensure that kernel sizes are positive integers
    weight_kernel = [max(1, k) for k in weight_kernel]
    bias_kernel = [max(1, k) for k in bias_kernel]

    # Ensure that strides are positive integers
    x_stride = max(1, x_stride)
    y_stride = max(1, y_stride)

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
                if (torch.any(parent_tensor < parent_mlp.parameter_range[0] + delta_abs) and delta < 0) or \
                     (torch.any(parent_tensor > parent_mlp.parameter_range[1] - delta_abs) and delta > 0):
                    continue
                
                #check if tensor is 2D (weights)
                if len(parent_tensor.shape) == 2:
                    new_weight_kernel = weight_kernel
                    new_x_stride = x_stride
                    new_y_stride = y_stride

                    # if the kernel is larger than the tensor itself in at least one dimension, we modify the kernel and strides accordingly
                    if (parent_tensor.shape[0] < weight_kernel[0] or parent_tensor.shape[1] < weight_kernel[1]):
                        # adjust kernel and strides
                        new_weight_kernel = [min(parent_tensor.shape[0], weight_kernel[0]), min(parent_tensor.shape[1], weight_kernel[1])]
                        min_dim = np.argmin(new_weight_kernel)
                        new_weight_kernel[min_dim] = max(1, new_weight_kernel[min_dim] // 2)  # halve (integer division) the smaller dimension of the kernel, ensure at least size 1
                        new_x_stride = new_weight_kernel[1]
                        new_y_stride = new_weight_kernel[0]

                    # sliding window over the tensor
                    for i in range(0, parent_tensor.shape[0] - new_weight_kernel[0] + 1, new_y_stride):
                        for j in range(0, parent_tensor.shape[1] - new_weight_kernel[1] + 1, new_x_stride):
                            neighbor_mlp = deepcopy(parent_mlp)
                            neighbor_model = neighbor_mlp.model
                            target_tensor = list(neighbor_model.parameters())[tensor_index].data
                            window = target_tensor[i:i+new_weight_kernel[0], j:j+new_weight_kernel[1]]
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

                # check if tensor is 1D (biases)
                elif len(parent_tensor.shape) == 1:
                    new_bias_kernel = bias_kernel
                    new_y_stride = y_stride
                    # if the kernel is larger than the tensor itself, we modify the kernel and stride accordingly
                    if parent_tensor.shape[0] < bias_kernel[0]:
                        new_bias_kernel = [min(parent_tensor.shape[0], bias_kernel[0])]
                        new_bias_kernel[0] = max(1, new_bias_kernel[0] // 2)  # halve (integer division) the kernel size, ensure at least size 1
                        new_y_stride = new_bias_kernel[0]
                    # sliding window over the tensor
                    for i in range(0, parent_tensor.shape[0] - new_bias_kernel[0] + 1, new_y_stride):
                        neighbor_mlp = deepcopy(parent_mlp)
                        neighbor_model = neighbor_mlp.model
                        target_tensor = list(neighbor_model.parameters())[tensor_index].data
                        window = target_tensor[i:i+new_bias_kernel[0]]
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

                else:
                    #raise an error in the neighborhood generation process if tensor is neither 1D nor 2D
                    raise ValueError(f"Unsupported tensor shape at index {tensor_index}: {parent_tensor.shape}. Only 1D and 2D tensors are supported.")
                
    return neighbors




def get_neighbors_layer_wise(search_node, X, Y, quantization_factor=None, 
                                     weight_kernels=None, bias_kernels=None, 
                                     weight_strides=None, bias_strides=None, 
                                     delta_abs=None):
    """
    Generates neighbor MLPs by applying perturbations using layer-specific kernels AND layer-specific strides.
    
    Args:
        weight_kernels (list of lists): A list containing the [h, w] kernel dimensions for each weight tensor.
                                        Example: [[2, 2], [3, 3]] (Height, Width)
        bias_kernels (list of lists):   A list containing the [length] kernel dimensions for each bias tensor.
                                        Example: [[2], [1]]
        weight_strides (list of lists): A list containing the [x_stride, y_stride] for each weight tensor.
                                        Example: [[1, 1], [2, 2]] (Horizontal step, Vertical step)
        bias_strides (list of lists):   A list containing the [stride] for each bias tensor.
                                        Example: [[1], [1]]
    """
    if quantization_factor is None:
        quantization_factor = search_node.quantized_mlp.quantization_factor

    # Basic validation to ensure lists are provided
    if weight_kernels is None or bias_kernels is None:
        raise ValueError("weight_kernels and bias_kernels must be provided as lists corresponding to the model layers.")
    
    if weight_strides is None or bias_strides is None:
        raise ValueError("weight_strides and bias_strides must be provided as lists corresponding to the model layers.")
    
    # Ensure that the kernels are valid (positive integers)
    for idx, kernel in enumerate(weight_kernels):
        kernel = [max(1, k) for k in kernel]  # Ensure kernel dimensions are at least 1
        weight_kernels[idx] = kernel

    for idx, kernel in enumerate(bias_kernels):
        kernel = [max(1, k) for k in kernel]  # Ensure kernel dimensions are at least 1
        bias_kernels[idx] = kernel
    
    # Ensure that the strides are valid (positive integers)
    for idx, stride in enumerate(weight_strides):
        stride = [max(1, s) for s in stride]  # Ensure strides are at least 1
        weight_strides[idx] = stride

    for idx, stride in enumerate(bias_strides):
        stride = [max(1, s) for s in stride]  # Ensure strides are at least 1
        bias_strides[idx] = stride

    neighbors = []
    parent_mlp = search_node.quantized_mlp
    parent_model = parent_mlp.model
    parent_parameter_list = list(parent_model.parameters())

    # Counters to track which specific layer configuration to use
    weight_idx = 0
    bias_idx = 0

    with torch.no_grad():
        # Iterate over all parameters in the model
        for tensor_index in range(len(parent_parameter_list)):
            
            # 1. Delta Setup
            if delta_abs is None:
                delta_abs = 1 / quantization_factor
            else:
                if (delta_abs * quantization_factor) % 1 != 0:
                    raise ValueError(f"delta_abs={delta_abs} is not compatible with quantization_factor={quantization_factor}.")
                
            for delta in [+delta_abs, -delta_abs]:
                parent_tensor = list(parent_model.parameters())[tensor_index].data

                # 2. Check for overflow
                if (torch.any(parent_tensor < parent_mlp.parameter_range[0] + delta_abs) and delta < 0) or \
                     (torch.any(parent_tensor > parent_mlp.parameter_range[1] - delta_abs) and delta > 0):
                    continue
                
                # 3. Handle 2D Tensors (Weights)
                if len(parent_tensor.shape) == 2:
                    # Retrieve the specific kernel and stride for this weight layer
                    try:
                        current_weight_kernel = weight_kernels[weight_idx]
                        current_weight_stride = weight_strides[weight_idx]
                    except IndexError:
                        raise IndexError(f"Not enough kernels or strides provided for weights. Index {weight_idx} out of range.")
                    
                    new_weight_kernel = list(current_weight_kernel) # Copy to avoid modifying input
                    
                    # Extract specific strides (Now format is [x_stride, y_stride])
                    new_x_stride = current_weight_stride[0]
                    new_y_stride = current_weight_stride[1]

                    # Dynamic shrinking: modify kernel/strides if kernel is larger than tensor
                    # Note: If the tensor is too small, we force the stride to follow the new (smaller) kernel size
                    if (parent_tensor.shape[0] < current_weight_kernel[0] or parent_tensor.shape[1] < current_weight_kernel[1]):
                        new_weight_kernel = [min(parent_tensor.shape[0], current_weight_kernel[0]), min(parent_tensor.shape[1], current_weight_kernel[1])]
                        min_dim = np.argmin(new_weight_kernel)
                        new_weight_kernel[min_dim] = max(1, new_weight_kernel[min_dim] // 2) 
                        
                        # Override user strides if shrinking occurred to ensure coverage
                        new_x_stride = new_weight_kernel[1]
                        new_y_stride = new_weight_kernel[0]

                    # Sliding window over the tensor
                    # Outer loop: Vertical (rows), step by y_stride
                    for i in range(0, parent_tensor.shape[0] - new_weight_kernel[0] + 1, new_y_stride):
                        # Inner loop: Horizontal (columns), step by x_stride
                        for j in range(0, parent_tensor.shape[1] - new_weight_kernel[1] + 1, new_x_stride):
                            neighbor_mlp = deepcopy(parent_mlp)
                            neighbor_model = neighbor_mlp.model
                            target_tensor = list(neighbor_model.parameters())[tensor_index].data
                            window = target_tensor[i:i+new_weight_kernel[0], j:j+new_weight_kernel[1]]
                            
                            window += delta

                            if quantization_factor is not None:
                                neighbor_mlp.quantization_factor = quantization_factor
                                neighbor_mlp.quantize()
                            else:
                                neighbor_mlp.quantize_tensor(tensor_index)
                            
                            if neighbor_mlp.overflow:
                                continue
                            
                            loss = neighbor_mlp.evaluate(X, Y)
                            neighbors.append((neighbor_mlp, loss))

                # 4. Handle 1D Tensors (Biases)
                elif len(parent_tensor.shape) == 1:
                    # Retrieve the specific kernel and stride for this bias layer
                    try:
                        current_bias_kernel = bias_kernels[bias_idx]
                        current_bias_stride = bias_strides[bias_idx]
                    except IndexError:
                        raise IndexError(f"Not enough kernels or strides provided for biases. Index {bias_idx} out of range.")

                    new_bias_kernel = list(current_bias_kernel)
                    
                    # Extract specific stride (assuming format [stride])
                    # Biases are 1D, so we just take the first element regardless of x/y naming convention
                    new_stride_1d = current_bias_stride[0]
                    
                    # Dynamic shrinking for bias
                    if parent_tensor.shape[0] < current_bias_kernel[0]:
                        new_bias_kernel = [min(parent_tensor.shape[0], current_bias_kernel[0])]
                        new_bias_kernel[0] = max(1, new_bias_kernel[0] // 2)
                        
                        # Override user stride if shrinking occurred
                        new_stride_1d = new_bias_kernel[0]
                    
                    # Sliding window over the tensor
                    for i in range(0, parent_tensor.shape[0] - new_bias_kernel[0] + 1, new_stride_1d):
                        neighbor_mlp = deepcopy(parent_mlp)
                        neighbor_model = neighbor_mlp.model
                        target_tensor = list(neighbor_model.parameters())[tensor_index].data
                        window = target_tensor[i:i+new_bias_kernel[0]]
                        
                        window += delta

                        if quantization_factor is not None:
                            neighbor_mlp.quantization_factor = quantization_factor
                            neighbor_mlp.quantize()
                        else:
                            neighbor_mlp.quantize_tensor(tensor_index)
                        
                        if neighbor_mlp.overflow:
                            continue
                        
                        loss = neighbor_mlp.evaluate(X, Y)
                        neighbors.append((neighbor_mlp, loss))

                else:
                    raise ValueError(f"Unsupported tensor shape at index {tensor_index}: {parent_tensor.shape}.")
            
            # Update Counters
            # We must increment the index counter corresponding to the tensor type found.
            # This happens outside the delta loop, but inside the tensor loop.
            tensor_shape_len = len(list(parent_model.parameters())[tensor_index].data.shape)
            if tensor_shape_len == 2:
                weight_idx += 1
            elif tensor_shape_len == 1:
                bias_idx += 1

    return neighbors




def get_neighbors_random(search_node, X, Y, quantization_factor=None, perturbation_ratio=0.1, search_coverage_ratio=0.5, delta_abs=None):
    """
    Generates neighbor MLPs by randomly sampling a subset of parameters within each layer 
    and perturbing them simultaneously.

    Args:
        perturbation_ratio (float): Between 0 and 1. The fraction of parameters in the current layer 
                                    to be selected for update in a single neighbor generation.
                                    (e.g., 0.1 means 10% of the weights in the layer are modified at once).
        search_coverage_ratio (float): Between 0 and 1. Determines the number of neighbors to generate 
                                       per layer, relative to the layer's total size.
                                       (e.g., 0.5 means we generate '0.5 * layer_size' different random masks).
    """
    
    if quantization_factor is None:
        quantization_factor = search_node.quantized_mlp.quantization_factor

    neighbors = []
    parent_mlp = search_node.quantized_mlp
    parent_model = parent_mlp.model
    parent_parameter_list = list(parent_model.parameters())

    with torch.no_grad():
        # Iterate over all layers (parameters)
        for tensor_index, parent_param in enumerate(parent_parameter_list):
            
            parent_tensor = parent_param.data
            num_elements = parent_tensor.numel()

            # 1. Determine Sample Size (How many params to change at once)
            subset_size = int(num_elements * perturbation_ratio)
            subset_size = max(1, subset_size) # Ensure at least 1 parameter is changed

            # 2. Determine Repetitions (How many random masks to try for this layer)
            num_trials = int(num_elements * search_coverage_ratio)
            num_trials = max(1, num_trials) # Ensure at least 1 trial is performed

            # 3. Delta Setup
            if delta_abs is None:
                delta_abs = 1 / quantization_factor
            else:
                if (delta_abs * quantization_factor) % 1 != 0:
                    raise ValueError(f"delta_abs={delta_abs} is not compatible with quantization_factor={quantization_factor}.")

            # 4. Neighborhood Generation Loop
            for _ in range(num_trials):
                
                # Select random indices for this trial (flattened indices)
                # We use torch.randperm to get unique indices without replacement
                indices = torch.randperm(num_elements)[:subset_size]

                for delta in [+delta_abs, -delta_abs]:
                    neighbor_mlp = deepcopy(parent_mlp)
                    neighbor_model = neighbor_mlp.model
                    
                    # Access the specific tensor in the copied model
                    target_param = list(neighbor_model.parameters())[tensor_index]
                    
                    # Create a flattened view to apply updates using the 1D indices
                    # (View shares memory, so modifying 'flat_target' modifies 'target_param')
                    flat_target = target_param.data.view(-1)
                    
                    # Check for overflow on the selected subset BEFORE applying
                    # Note: We check the PARENT tensor values to see if adding delta would overflow limits
                    parent_flat = parent_tensor.view(-1)
                    subset_values = parent_flat[indices]

                    if (torch.any(subset_values < parent_mlp.parameter_range[0] + delta_abs) and delta < 0) or \
                       (torch.any(subset_values > parent_mlp.parameter_range[1] - delta_abs) and delta > 0):
                        continue

                    # Apply delta to the selected subset
                    flat_target[indices] += delta

                    # Quantization handling
                    if quantization_factor is not None:
                        neighbor_mlp.quantization_factor = quantization_factor
                        neighbor_mlp.quantize()
                    else:
                        neighbor_mlp.quantize_tensor(tensor_index)
                    
                    if neighbor_mlp.overflow:
                        continue
                    
                    loss = neighbor_mlp.evaluate(X, Y)
                    neighbors.append((neighbor_mlp, loss))

    return neighbors
