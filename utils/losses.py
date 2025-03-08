import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import diffdist
from utils.gather import GatherLayer

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, device="cuda"):
        super().__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool, device = self.device)
        mask = mask.fill_diagonal_(0) # self-similarity is not useful
        for i in range(batch_size):
            mask[i, batch_size + i] = 0 # mask the positive pair
            mask[batch_size + i, i] = 0

        # register the mask as a buffer
        self.register_buffer("mask", mask)
        return mask

    def forward(self, z_i, z_j):
        # distributed version
        if dist.is_initialized():
            z_i = torch.cat(GatherLayer.apply(z_i), dim=0)
            z_j = torch.cat(GatherLayer.apply(z_j), dim=0)
        
        z = torch.cat([z_i, z_j], dim=0)
        
        N = z.size(0)
        self.batch_size = N // 2
        # Mask correlated samples (positive and self pairs)
        self.mask = self.mask_correlated_samples(self.batch_size)

        # Compute the NxN similarity matrix (cosine similarity)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # Extract the positive samples
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # Concatenate positive samples (for each view i and j)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        # Extract negative samples by masking out the positive pairs
        negative_samples = sim[self.mask].reshape(N, -1)  # Masking positive pairs

        # Labels for cross-entropy loss (all zeros for correct classification)
        labels = torch.zeros(N).to(positive_samples.device).long()

        # Concatenate positive and negative samples for logits
        logits = torch.cat((positive_samples, negative_samples), dim=1)  # N x (N-1)

        # Calculate the loss using cross-entropy
        loss = self.criterion(logits, labels)
        loss /= N  # Normalize by batch size

        return loss

class LossFactory:
    @staticmethod
    def get_loss(loss_name, **kwargs):
        if loss_name == "simclr":
            return NTXentLoss(**kwargs)
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
