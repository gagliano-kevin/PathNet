
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