import torch
import torch.nn as nn



class WineMLP(nn.Module):

    def __init__(self, input_size=11, hidden_size_1=32, hidden_size_2=32, output_size=6):
        super(WineMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, output_size)
        )
        
    def forward(self, x):
        return self.net(x)



class HousingMLP(torch.nn.Module):

    def __init__(self, input_size=8, hidden_size_1=32, hidden_size_2=32, output_size=1):
        super(HousingMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, output_size)
        )

    def forward(self, x):
        return self.net(x)



class IrisMLP(nn.Module):

    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, num_classes):
        super().__init__()
        # 4 features -> Hidden Layer 1
        self.fc1 = nn.Linear(input_size, hidden_size_1) 
        # Hidden Layer 1 -> Hidden Layer 2
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        # Hidden Layer 2 -> Hidden Layer 3
        self.fc3 = nn.Linear(hidden_size_2, hidden_size_3)
        # Hidden Layer 3 -> 3 classes
        self.fc4 = nn.Linear(hidden_size_3, num_classes)
        
    def forward(self, x):
        # ReLU activation for hidden layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        # No activation on the output layer when using nn.CrossEntropyLoss
        # (it handles the softmax internally for better numerical stability)
        out = self.fc4(x)
        return out



class SinusoidalMLP(nn.Module):
    
    def __init__(self, input_size=1, hidden_size_1=32, hidden_size_2=32, hidden_size_3=32, output_size=1):
        super(SinusoidalMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, hidden_size_3),
            nn.ReLU(),
            nn.Linear(hidden_size_3, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)
    