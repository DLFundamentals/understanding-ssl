import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import BaseEncoder, ResNetEncoder
from models.projector import SimCLR_Projector

from utils.metrics import KNN
from utils.analysis import cal_cdnv, embedding_performance_nearest_mean_classifier

import os
from tqdm import tqdm
from collections import defaultdict
import wandb

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
            elif 'cifar' in dataset:
                h = self.encoder(torch.randn(1, 3, 32, 32))
            elif dataset == 'svhn':
                h = self.encoder(torch.randn(1, 3, 32, 32))
            else:
                raise NotImplementedError(f"{dataset} not implemented")
            
        input_dim = h.shape[1]
        
        self.projector = SimCLR_Projector(input_dim, hidden_dim, projection_dim)

        # whether to track performance or not
        self.track_performance = kwargs.get('track_performance', False)

    def forward(self, X):
  
        h = self.encoder(X)
        h = h.view(h.size(0), -1) # flatten the tensor
        g_h = self.projector(h)

        return h, F.normalize(g_h, dim = -1)
    
    # ========== Training Function ==========
    def custom_train(self, train_loader,
              criterion, optimizer, num_epochs, 
              augment_both = True, save_every = 10, log_every=10,
              experiment_name = 'simclr/cifar10',
              device = 'cuda', **kwargs):
        
        self.to(device) # move model to device
        print(f"Training on {device} started! Experiment name: {experiment_name}")

        # useful for logging
        self.optimizer = optimizer
        self.wandb_defined = False

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

            if (epoch + 1) % log_every == 0:
                # Evaluate
                if self.track_performance:
                    with torch.no_grad():
                        outputs = self.custom_eval(train_loader)
                    
                    self.log_metrics(outputs, epoch, avg_loss)


        print("Training Complete! 🎉")

    # ========== Evaluation Function ==========
    def custom_eval(self, train_loader, test_loader=None, 
                    **kwargs):
        self.eval()
        print("Evaluation Started!")

        perform_knn = kwargs.get('perform_knn', False)
        perform_cdnv = kwargs.get('perform_cdnv', False)
        perform_nccc = kwargs.get('perform_nccc', False)
        settings = kwargs.get('settings', None)

        outputs = defaultdict(list)

        if perform_knn:
            # Evaluate KNN
            knn_evaluator = KNN(self, self.K)
            train_acc, test_acc = knn_evaluator.knn_eval(train_loader, test_loader)
            outputs['knn_train_acc'].append(train_acc)
            outputs['knn_test_acc'].append(test_acc)

        if perform_cdnv:
            # Evaluate CDN-V
            cdnvs = cal_cdnv(self, settings, train_loader)
            outputs['cdnv'] = cdnvs

        if perform_nccc:
            # Evaluate NCCC
            nccc = embedding_performance_nearest_mean_classifier(self, settings, train_loader)
            outputs['nccc'] = nccc
            
        # print("Evaluation Complete! 🎉")
        return outputs

    # ========== Run One Batch =================
    def run_one_batch(self, batch, criterion, optimizer=None, device='cuda'):
        # get the inputs
        view1, view2, labels = batch
        # skip the batch with only 1 image
        if view1.size(0) < 2:
            return 0
        view1, view2 = view1.to(device), view2.to(device)
        labels = labels.to(device)

        # forward pass
        view1_features, view1_proj = self(view1)
        view2_features, view2_proj = self(view2)

        # compute contrastive loss
        loss = criterion(view1_proj, view2_proj, labels)

        return loss
    
    def log_metrics(self, eval_outputs, cur_epoch, cur_loss_per_epoch):
        # define epoch as x-axis
        if not self.wandb_defined:
            wandb.define_metric("epoch")
            wandb.define_metric("learning_rate", step_metric="epoch")
            wandb.define_metric("knn_accuracy", step_metric="epoch")
            wandb.define_metric("cdnv_0", step_metric="epoch")
            wandb.define_metric("log_cdnv_0", step_metric="epoch")
            wandb.define_metric("cdnv_1", step_metric="epoch")
            wandb.define_metric("log_cdnv_1", step_metric="epoch")
            wandb.define_metric("nccc_0", step_metric="epoch")
            wandb.define_metric("nccc_1", step_metric="epoch")

            # define loss per epoch
            wandb.define_metric("loss_per_epoch", step_metric="epoch")
            self.wandb_defined = True

        # collect all logs in one dictionary
        log_data = {"epoch": cur_epoch}
        log_data["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        log_data["loss_per_epoch"] = cur_loss_per_epoch

        if self.perform_knn:
            log_data["knn_accuracy"] = eval_outputs["knn_train_acc"]

        if self.perform_cdnv:
            if isinstance(eval_outputs["cdnv"], list):
                for i, cdnv in enumerate(eval_outputs["cdnv"]):
                    log_data[f'cdnv_{i}'] = cdnv
                    log_data[f'log_cdnv_{i}'] = torch.log10(torch.tensor(cdnv))
            else:
                log_data["cdnv_0"] = eval_outputs["cdnv"]
                log_data["log_cdnv_0"] = torch.log10(torch.tensor(eval_outputs["cdnv"]))

        if self.perform_nccc:
            if isinstance(eval_outputs["nccc"], list):
                for i, nccc in enumerate(eval_outputs["nccc"]):
                    log_data[f'nccc_{i}'] = nccc
            else:
                log_data["nccc"] = eval_outputs["nccc"]

        # log all metrics
        wandb.log(log_data)