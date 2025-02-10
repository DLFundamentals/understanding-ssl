import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import BaseEncoder, ResNetEncoder
from models.projector import SimCLR_Projector

from utils.metrics import KNN

import os
from tqdm import tqdm

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

        # whether to track performance or not
        self.track_performance = kwargs.get('track_performance', False)
        if self.track_performance:
            self.K = kwargs.get('K', 5)

    def forward(self, X):
  
        h = self.encoder(X)
        h = h.view(h.size(0), -1) # flatten the tensor
        g_h = self.projector(h)

        return h, F.normalize(g_h, dim = -1)
    
    # ========== Training Function ==========
    def custom_train(self, train_loader,
              criterion, optimizer, num_epochs, 
              augment_both = True, save_every = 10, 
              experiment_name = 'simclr/cifar10',
              device = 'cuda', **kwargs):
        
        self.to(device) # move model to device
        print(f"Training on {device} started! Experiment name: {experiment_name}")

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0

            for batch in tqdm(train_loader):
                
                # get the inputs
                view1, view2, _ = batch
                # skip the batch with only 1 image
                if view1.size(0) < 2:
                    continue
                view1, view2 = view1.to(device), view2.to(device)

                # forward pass
                view1_features, view1_proj = self(view1)
                view2_features, view2_proj = self(view2)

                # compute contrastive loss
                loss = criterion(view1_proj, view2_proj)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs} Loss: {avg_loss:.4f}")

            # Save Model & Logs
            if (epoch + 1) % save_every == 0:
                checkpoint_path = f"experiments/{experiment_name}/checkpoints/epoch_{epoch+1}.pth"
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(self.state_dict(), checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

                # Evaluate KNN
                if self.track_performance:
                    self.custom_eval(train_loader)


        print("Training Complete! 🎉")

    # ========== Evaluation Function ==========
    def custom_eval(self, train_loader, test_loader=None):
        # Evaluate KNN
        knn_evaluator = KNN(self, self.K)
        train_acc, test_acc = knn_evaluator.knn_eval(train_loader, test_loader)

        # print(f"KNN Evaluation: Train Acc: {train_acc:.2f}%")
        # if test_acc:
        #     print(f"KNN Evaluation: Test Acc: {test_acc:.2f}%")
        # print("Evaluation Complete! 🎉")