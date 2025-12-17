#===================================================================================================================================
#===================================================================================================================================
#-------------------------------------- run this file from project root: python -m test --------------------------------------------
#===================================================================================================================================
#===================================================================================================================================

import torch
import torch.nn as nn
from copy import deepcopy




# test for checking if cloned model parameters are independent of the original model parameters
"""class MyModel:
    def __init__(self):
        self.net = nn.Sequential(nn.Linear(10, 10))

original = MyModel()
cloned = deepcopy(original)

# Check if the memory addresses of the underlying data are different
original_ptr = original.net[0].weight.data_ptr()
cloned_ptr = cloned.net[0].weight.data_ptr()

print(f"Original Pointer: {original_ptr}")
print(f"Cloned Pointer:   {cloned_ptr}")
print(f"Are they independent? {original_ptr != cloned_ptr}")"""




model = nn.Sequential(
    nn.Linear(2, 4),  
    nn.ReLU(),
    nn.Linear(4, 2)   
)


print("\n\ntype(model)\n",type(model))
print("\n\nmodel\n",model)

print("\n\ntype(model.parameters())\n",type(model.parameters()))
print("\n\nmodel.parameters()\n",model.parameters())

parameter_list = list(model.parameters())
print("\n\ntype(parameter_list)\n",type(parameter_list))
print("\n\nlen(parameter_list)\n",len(parameter_list))
print("\n\nparameter_list\n",parameter_list)

print("\n\ntype(parameter_list[0])\n",type(parameter_list[0]))
print("\n\nparameter_list[0].shape\n",parameter_list[0].shape)
print("\n\nparameter_list[0]\n",parameter_list[0])

print("\n\ntype(parameter_list[0]).data\n",type(parameter_list[0].data))
print("\n\nparameter_list[0].data\n",parameter_list[0].data)

print("\n\ntype(parameter_list[1])\n",type(parameter_list[1]))
print("\n\nparameter_list[1].shape\n",parameter_list[1].shape)
print("\n\nparameter_list[1]\n",parameter_list[1])

print("\n\ntype(parameter_list[1]).data\n",type(parameter_list[1].data))
print("\n\nparameter_list[1].data\n",parameter_list[1].data)



#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#               simulating neighborhood generation 
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

#defining two kernels (or sliding windows) for demonstration
weight_kernel = [2,2]  # 2D kernel of size 2x2 for weights
bias_kernel = [2]      # 1D kernel of size 2 for biases

# defining a small increment value
delta_abs = 0.01         # small value to add for demonstration

# parent model parameters
parent_model = model
parent_parameter_list = list(parent_model.parameters())

#let's iterate over all parameters
for tensor_index in range(len(parameter_list)):

    print(f"\n\nOriginal Tensor at index {tensor_index}:\n", parent_parameter_list[tensor_index].data)

    for delta in [+delta_abs, -delta_abs]:

        neighbor_model = deepcopy(parent_model)
        neighbor_parameter_list = list(neighbor_model.parameters())
        original_tensor = neighbor_parameter_list[tensor_index].data

        #check if tensor is 2D (weights)
        if len(original_tensor.shape) == 2:
            # check if the tensor is compatible with the weight kernel (has at least the size of the kernel)
            if original_tensor.shape[0] >= weight_kernel[0] and original_tensor.shape[1] >= weight_kernel[1]:
                print(f"\n\nProcessing 2D weight tensor at index {tensor_index} with shape {original_tensor.shape}")
                # sliding window over the tensor
                for i in range(original_tensor.shape[0] - weight_kernel[0] + 1):
                    for j in range(original_tensor.shape[1] - weight_kernel[1] + 1):
                        window = original_tensor[i:i+weight_kernel[0], j:j+weight_kernel[1]]
                        print(f"Weight Window starting at ({i},{j}):")
                        print(window)

                        #adding a small delta to each element in the window for demonstration
                        window += delta
                        print(f"Modified Weight Window starting at ({i},{j}):")
                        print(window)
            # else the kernel is larger than the tensor itself and we take the whole tensor
            else:
                print(f"\n\n2D weight tensor at index {tensor_index} is smaller than kernel, taking whole tensor:")
                print(original_tensor)

                #adding a small delta to each element in the tensor for demonstration
                original_tensor += delta
                print(f"Modified 2D weight tensor at index {tensor_index}:")

        # check if tensor is 1D (biases)
        elif len(original_tensor.shape) == 1:
            # check if the tensor is compatible with the bias kernel (has at least the size of the kernel)
            if original_tensor.shape[0] >= bias_kernel[0]:
                print(f"\n\nProcessing 1D bias tensor at index {tensor_index} with shape {original_tensor.shape}")
                # sliding window over the tensor
                for i in range(original_tensor.shape[0] - bias_kernel[0] + 1):
                    window = original_tensor[i:i+bias_kernel[0]]
                    print(f"Bias Window starting at ({i}):")
                    print(window)

                    #adding a small delta to each element in the window for demonstration
                    window += delta
                    print(f"Modified Bias Window starting at ({i}):")
                    print(window)
            # else the kernel is larger than the tensor itself and we take the whole tensor
            else:
                print(f"\n\n1D bias tensor at index {tensor_index} is smaller than kernel, taking whole tensor:")
                print(original_tensor)

                #adding a small delta to each element in the tensor for demonstration
                original_tensor += delta
                print(f"Modified 1D bias tensor at index {tensor_index}:")
                print(original_tensor)

        else:
            #raise an error in the neighborhood generation process if tensor is neither 1D nor 2D
            raise ValueError(f"Unsupported tensor shape at index {tensor_index}: {original_tensor.shape}. Only 1D and 2D tensors are supported.")





