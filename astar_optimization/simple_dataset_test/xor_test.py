#===================================================================================================================================
#===================================================================================================================================
#-------------- run this file from project root: python -m astar_optimization.simple_dataset_test.xor_test -------------------------
#===================================================================================================================================
#===================================================================================================================================

import torch
import torch.nn as nn
from source.PathNet import Trainer

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


model = nn.Sequential(
nn.Linear(2, 2),
nn.ReLU(),
nn.Linear(2, 1),
nn.Sigmoid()
) 

trainer = Trainer(model, nn.MSELoss(), quantization_factor=2, parameter_range=(-4, 4), debug_mlp=True, param_fraction=1.0, max_iterations=2000, log_freq=1000, target_loss=0.0001, update_strategy=2, g_ini_val=0, g_step=0.01, alpha=0.5, scale_f=True)

trainer.train(XOR_DATASET, XOR_LABELS)

trainer.plot_training_history('xor_base.png')

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