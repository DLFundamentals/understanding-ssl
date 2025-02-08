import torch
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# utils
from utils.augmentations import get_transforms
from utils.dataset_loader import get_dataset
from utils.losses import NTXentLoss

# model
from models.simclr import SimCLR

import argparse
import yaml
from tqdm import tqdm

# set seed
torch.manual_seed(123)
torch.cuda.manual_seed(123)


# ========== Training Function ==========
def train(model, train_loader,
          criterion, optimizer, num_epochs, augment_both = False,
          save_every=50, experiment_name="simclr/cifar10",
          device='cuda'):
    """Runs the training loop for self-supervised learning models."""
    print("Training Started!")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader):
            view1_images, view2_images = batch
            # Skip batches with only 1 image, could be the last batch
            if len(view1_images) < 2:
                continue

            # Move to device
            view1_images = view1_images.to(device)
            view2_images = view2_images.to(device)
            
            # Forward Pass
            view1_features, view1_projections = model(view1_images)
            view2_features, view2_projections = model(view2_images)

            # Compute contrastive loss
            loss = criterion(view1_projections, view2_projections)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Save Model & Logs
        if (epoch + 1) % save_every == 0:
            checkpoint_path = f"experiments/{experiment_name}/checkpoints/epoch_{epoch+1}.pth"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    print("Training Complete! 🎉")


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='SimCLR Training')
    parser.add_argument('--config', '-c', required=True, help='path to yaml config file')
    args = parser.parse_args()

    # load config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # load config parameters
    experiment_name = config['experiment_name']
    method_type = config['method_type']

    dataset_name = config['dataset']['name']
    dataset_path = config['dataset']['path']
    
    batch_size = config['training']['batch_size']
    epochs = config['training']['num_epochs']
    lr = config['training']['lr']
    augmentations_type = config['training']['augmentations_type'] # imagenet or cifar or other dataset name
    augment_both = config['training']['augment_both']
    save_every = config['training']['save_every']

    encoder_type = config['model']['encoder_type']
    width_multiplier = config['model']['width_multiplier']
    projection_dim = config['model']['projection_dim']
    hidden_dim = config['model']['hidden_dim']

    temperature = config['loss']['temperature']

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load dataset
    _, train_loader = get_dataset(dataset_name=dataset_name, 
                                dataset_path=dataset_path,
                                augment_both_views=augment_both,
                                batch_size=batch_size,)
    
    # train_transforms, basic_transforms = get_transforms(dataset=augmentations_type)

    # define model
    if encoder_type == 'resnet50':
        encoder = torchvision.models.resnet50(pretrained=False)
    else:
        raise NotImplementedError(f"{encoder_type} not implemented")
    
    if method_type == 'simclr':
        ssl_model = SimCLR(model=encoder,
                           dataset=dataset_name,
                           width_multiplier=width_multiplier,
                           hidden_dim=hidden_dim,
                           projection_dim=projection_dim)
    else:
        raise NotImplementedError(f"{method_type} not implemented")
    
    ssl_model = ssl_model.to(device)

    # define loss & optimizer
    criterion = NTXentLoss(batch_size, temperature, device)
    optimizer = optim.Adam(ssl_model.parameters(), lr=lr) # replace with LARS for large batch sizes

    # train model
    ssl_model.custom_train(train_loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    num_epochs=epochs,
                    augment_both=augment_both,
                    save_every=save_every,
                    experiment_name=experiment_name,
                    device=device)