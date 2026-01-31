import torch
import torch.nn as nn

class BaseDynamicMLP(nn.Module):
    """A helper base class to avoid repeating the building logic."""
    def __init__(self, input_size, hidden_layers, output_size, final_activation=None):
        super(BaseDynamicMLP, self).__init__()
        
        layers = []
        in_dim = input_size
        
        # Build hidden layers
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, output_size))
        
        # Optional final activation (like Tanh for Sinusoidal)
        if final_activation:
            layers.append(final_activation)
            
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --- Specific Dataset Classes ---

class WineMLP(BaseDynamicMLP):
    def __init__(self, input_size=11, hidden_layers=[32, 32], output_size=6):
        # Note: Wine uses CrossEntropyLoss, so no final activation here
        super().__init__(input_size, hidden_layers, output_size)

class HousingMLP(BaseDynamicMLP):
    def __init__(self, input_size=8, hidden_layers=[64, 32, 16], output_size=1):
        super().__init__(input_size, hidden_layers, output_size)

class IrisMLP(BaseDynamicMLP):
    def __init__(self, input_size=4, hidden_layers=[16, 16], output_size=3):
        # Note: Iris uses CrossEntropyLoss, so no final activation here
        super().__init__(input_size, hidden_layers, output_size)

class SinusoidalMLP(BaseDynamicMLP):
    def __init__(self, input_size=1, hidden_layers=[32, 32, 32], output_size=1):
        # Sinusoidal often uses Tanh to bound the output range
        super().__init__(input_size, hidden_layers, output_size, final_activation=nn.Tanh())