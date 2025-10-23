#===================================================================================================================================
#===================================================================================================================================
#----------------- run this file from project root: python -m optimizations_comparison.sine_comparison -----------------------------
#===================================================================================================================================
#===================================================================================================================================

from source.sinusoidal_func_utils import generate_sinusoidal_tensor, plot_sine_predictions, SinCosDataset, SinusoidalMLP
from source.general_utils import plot_losses
from source.PathNet import Trainer

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


NUM_SAMPLES = 1000
MIN_ANGLE = 0
MAX_ANGLE = 4 * np.pi
NOISE_LEVEL = 0.1
ITERATIONS = 100

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- ASTAR TRAINING -----------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

X_sin, Y_sin = generate_sinusoidal_tensor(func=torch.sin, num_samples=NUM_SAMPLES, min_angle=MIN_ANGLE, max_angle=MAX_ANGLE, noise_level=NOISE_LEVEL)

model = nn.Sequential(
    nn.Linear(1, 6),  
    nn.ReLU(),
    #nn.Tanh(),
    nn.Linear(6, 6),
    nn.ReLU(),
    #nn.Tanh(),
    nn.Linear(6, 1),
    #nn.Tanh()
)

trainer = Trainer(model, nn.MSELoss(), quantization_factor=2, parameter_range=(-5, 5), debug_mlp=True, param_fraction=1.0, max_iterations=ITERATIONS, log_freq=100, target_loss=0.01, update_strategy=2, g_ini_val=0, g_step=0.01, alpha=0.5, scale_f=True)

trainer.train(X_sin, Y_sin)

plot_sine_predictions(test_x_np=X_sin.numpy(), 
                      predicted_sin_np=trainer.best_node.quantized_mlp.model(X_sin).detach().numpy(), 
                      true_sin_np=Y_sin.numpy(),
                      filename="sine_model_astar.png")



#------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------- GRADIENT BASE TRAINING ---------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------


BATCH_SIZE = NUM_SAMPLES    # Full batch
LEARNING_RATE = 0.001
EPOCHS = ITERATIONS
HIDDEN_SIZE = 64

dataset = SinCosDataset(NUM_SAMPLES, MIN_ANGLE, MAX_ANGLE, NOISE_LEVEL)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    
sin_model = SinusoidalMLP(hidden_size=HIDDEN_SIZE)
criterion = nn.MSELoss()
sin_optimizer = torch.optim.Adam(sin_model.parameters(), lr=LEARNING_RATE)

loss_history = []

print("Starting training for Sine model...")
for epoch in range(EPOCHS):
    total_loss = 0
    for x_batch, sin_y_batch, cos_y_batch in dataloader:    # Only using sin_y_batch for sine model
        sin_optimizer.zero_grad()
        predictions = sin_model(x_batch)
        loss = criterion(predictions, sin_y_batch) # Target is sin_y_batch
        loss.backward()
        sin_optimizer.step()
        total_loss += loss.item()
    loss_history.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Sine Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(dataloader):.6f}')
print("Sine model training finished.")

plot_sine_predictions(test_x_np=dataset.x_data.numpy(), 
                      predicted_sin_np=sin_model(dataset.x_data).detach().numpy(), 
                      true_sin_np=dataset.sin_y_data.numpy(),
                      filename="sine_model_grad_base.png")

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------


losses = [trainer.loss_history, loss_history]
loss_labels = ["A-Star", "Gradient base"]

plot_losses(losses, loss_labels, "sine_loss_comparison.png")