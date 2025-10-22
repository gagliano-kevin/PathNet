#===================================================================================================================================
#===================================================================================================================================
#--------------------- run this file from project root: python -m gradient_optimization.sine_grad_base -----------------------------
#===================================================================================================================================
#===================================================================================================================================

from source.sinusoidal_func_utils import generate_sinusoidal_tensor, plot_sine_predictions
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# --- Hyperparameters for the data ---
NUM_SAMPLES = 1000
MIN_ANGLE = 0
MAX_ANGLE = 4 * np.pi
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 500
HIDDEN_SIZE = 64

class SinCosDataset(Dataset):
    def __init__(self, num_samples, min_angle, max_angle, noise_level=0.1):
        self.x_data = torch.linspace(min_angle, max_angle, num_samples).float().unsqueeze(1)
        
        sin_y = torch.sin(self.x_data)
        cos_y = torch.cos(self.x_data)
        
        # Add noise
        noise = torch.randn_like(sin_y) * noise_level
        self.sin_y_data = sin_y + noise
        self.cos_y_data = cos_y + noise

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        # Returns x, sin(x), cos(x)
        return self.x_data[idx], self.sin_y_data[idx], self.cos_y_data[idx]

# Create the dataset
dataset = SinCosDataset(NUM_SAMPLES, MIN_ANGLE, MAX_ANGLE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

class SinusoidalMLP(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(SinusoidalMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)
    
sin_model = SinusoidalMLP(hidden_size=HIDDEN_SIZE)
criterion = nn.MSELoss()
sin_optimizer = torch.optim.Adam(sin_model.parameters(), lr=LEARNING_RATE)

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

    if (epoch + 1) % 10 == 0:
        print(f'Sine Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(dataloader):.6f}')
print("Sine model training finished.")

plot_sine_predictions(test_x_np=dataset.x_data.numpy(), 
                      predicted_sin_np=sin_model(dataset.x_data).detach().numpy(), 
                      true_sin_np=dataset.sin_y_data.numpy(),
                      filename="sine_model_grad_base.png")