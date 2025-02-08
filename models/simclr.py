import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import BaseEncoder, ResNetEncoder
from models.projector import SimCLR_Projector

class SimCLR(nn.Module):
    def __init__(self, model, layer = -2, dataset = 'imagenet',
                 width_multiplier = 1, pretrained = False,
                 hidden_dim = 512, projection_dim = 128, **kwargs):
        
        super().__init__()

        self.encoder = ResNetEncoder(model, layer = layer, dataset = dataset,
                                     width_multiplier = width_multiplier,
                                     pretrained = pretrained)
        
        # run a mock image tensor to instantiate parameters
        with torch.no_grad():
            if dataset == 'imagenet':
                h = self.encoder(torch.randn(1, 3, 224, 224))
            elif dataset == 'cifar' or 'cifar' in dataset:
                h = self.encoder(torch.randn(1, 3, 32, 32))
            else:
                raise NotImplementedError(f"{dataset} not implemented")
            
        input_dim = h.shape[1]
        
        self.projector = SimCLR_Projector(input_dim, hidden_dim, projection_dim)

    def forward(self, X):
  
        h = self.encoder(X)
        h = h.view(h.size(0), -1) # flatten the tensor
        g_h = self.projector(h)

        return h, F.normalize(g_h, dim = -1)