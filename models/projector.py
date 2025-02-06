import torch
import torch.nn as nn

class SimCLR_Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, projection_dim=128):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, X):
        return self.projector(X)