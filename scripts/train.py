import torch
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchlars import LARS

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
    track_performance = config['training']['track_performance']
    K = config['evaluation']['K'] if track_performance else None


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
                           projection_dim=projection_dim,
                           track_performance=track_performance,
                            K=K)
    else:
        raise NotImplementedError(f"{method_type} not implemented")
    
    ssl_model = ssl_model.to(device)

    # define loss & optimizer
    criterion = NTXentLoss(batch_size, temperature, device)
    # optimizer = optim.Adam(ssl_model.parameters(), lr=lr) # replace with LARS for large batch sizes

    # ========= LARS optimizer =========
    base_optimizer = optim.SGD(ssl_model.parameters(), lr=lr, momentum=0.9)
    optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)

    # train model
    ssl_model.custom_train(train_loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    num_epochs=epochs,
                    augment_both=augment_both,
                    save_every=save_every,
                    experiment_name=experiment_name,
                    device=device,)