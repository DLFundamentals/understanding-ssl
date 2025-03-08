import torch
import torch.nn as nn

class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False

class SimCLR_Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, projection_dim=128):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim, bias=False),
            BatchNorm1dNoBias(projection_dim)
        ) # we do not have any activation function after the last layer

    def forward(self, X):
        return self.projector(X)