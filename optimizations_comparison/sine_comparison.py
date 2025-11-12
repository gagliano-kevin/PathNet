#===================================================================================================================================
#===================================================================================================================================
#----------------- run this file from project root: python -m optimizations_comparison.sine_comparison -----------------------------
#===================================================================================================================================
#===================================================================================================================================

from source.sinusoidal_func_utils import generate_sinusoidal_tensor, plot_sine_predictions, SinCosDataset, SinusoidalMLP, SinusoidalMLP_tanh_out
from source.general_utils import plot_losses
from source.SimplePathNet import Trainer

import time

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


NUM_SAMPLES = 1000
MIN_ANGLE = 0
MAX_ANGLE = 4 * np.pi
NOISE_LEVEL = 0.1
ITERATIONS = 1000

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

xs_model = nn.Sequential(
    nn.Linear(1, 4),  
    nn.ReLU(),
    #nn.Tanh(),
    nn.Linear(4, 4),
    nn.ReLU(),
    #nn.Tanh(),
    nn.Linear(4, 1),
    nn.Tanh()
    )

trainer = Trainer(xs_model, nn.MSELoss(), quantization_factor=10, parameter_range=(-10, 10), debug_mlp=True, param_fraction=1.0, max_iterations=ITERATIONS, log_freq=100, target_loss=0.01, measure_time=True)

trainer.train(X_sin, Y_sin)

trainer.log_to_file("sine_model_astar_1k_iters_log.txt")

plot_sine_predictions(test_x_np=X_sin.numpy(), 
                      predicted_sin_np=trainer.best_node.quantized_mlp.model(X_sin).detach().numpy(), 
                      true_sin_np=Y_sin.numpy(),
                      filename="sine_model_astar_1k_iters.png")


#------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------- GRADIENT BASE TRAINING ---------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------


BATCH_SIZE = NUM_SAMPLES    # Full batch
LEARNING_RATE = 0.001
EPOCHS = ITERATIONS
HIDDEN_SIZE = 4

dataset = SinCosDataset(NUM_SAMPLES, MIN_ANGLE, MAX_ANGLE, NOISE_LEVEL)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    
sin_model = SinusoidalMLP(hidden_size=HIDDEN_SIZE)
sin_model_tanh_out = SinusoidalMLP_tanh_out(hidden_size=HIDDEN_SIZE)
criterion = nn.MSELoss()
sin_optimizer = torch.optim.Adam(sin_model.parameters(), lr=LEARNING_RATE)

loss_history = []

start_time = time.time()

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


end_time = time.time()
training_time = end_time - start_time
print(f"Sine model training time: {training_time:.2f} seconds")

with open("sine_model_grad_base_1k_iters_log.txt", "w") as f:
    for i, loss in enumerate(loss_history):
        f.write(f"Iteration {i+1}: Loss = {loss}\n")
    f.write(f"\n\nTotal training time (seconds): {training_time:.2f}\n")

print("Sine model training finished.")

plot_sine_predictions(test_x_np=dataset.x_data.numpy(), 
                      predicted_sin_np=sin_model(dataset.x_data).detach().numpy(), 
                      true_sin_np=dataset.sin_y_data.numpy(),
                      filename="sine_model_grad_base_1k_iters.png")


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

losses = [trainer.loss_history, loss_history]
loss_labels = ["A-Star", "Gradient base"]

plot_losses(losses, loss_labels, "sine_loss_comparison_1k_iters.png")