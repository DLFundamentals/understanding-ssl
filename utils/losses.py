import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, device="cuda"):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.mask = self.mask_correlated_samples(batch_size)
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

        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0) # concatenate the two batches (two views) for a total of 2N samples

        # Compute the NxN similarity matrix
        sim = self.similarity_f(z.unsqueeze(1), # N x 1 x D
                                z.unsqueeze(0)  # 1 x N x D
                                ) / self.temperature 
        
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1) # N x (N-2)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1) # N x N-1
        loss = self.criterion(logits, labels)
        loss /= N

        return loss

class LossFactory:
    @staticmethod
    def get_loss(loss_name, **kwargs):
        if loss_name == "simclr":
            return NTXentLoss(**kwargs)
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
