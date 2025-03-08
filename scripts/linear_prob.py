import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# utils
from utils.dataset_loader import get_dataset
from utils.analysis import embedding_performance

# model
from models.simclr import SimCLR

import argparse
import yaml
from tqdm import tqdm
from collections import namedtuple
import wandb

# set seed
torch.manual_seed(123)
torch.cuda.manual_seed(123)

def load_model(ssl_model, config, ckpt_path):
    if ckpt_path is None:
        ckpt_path = config['model']['ckpt_path']

    if ckpt_path is None:
        raise ValueError("ckpt_path not provided")
    
    snapshot = torch.load(ckpt_path, map_location='cuda', weights_only=True)
    ssl_model.load_state_dict(snapshot['MODEL_STATE'])
    print(f"Model loaded from {ckpt_path}")


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='SimCLR Linear Probing')
    parser.add_argument('--config', '-c', required=True, help='path to yaml config file')
    parser.add_argument('--ckpt_path', '-ckpt', 
                        default=None,
                        help='path to model checkpoint')
    args = parser.parse_args()

    # load config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # load config parameters
    experiment_name = config['experiment_name']
    method_type = config['method_type']

    dataset_name = config['dataset']['name']
    dataset_path = config['dataset']['path']

    encoder_type = config['model']['encoder_type']
    width_multiplier = config['model']['width_multiplier']
    hidden_dim = config['model']['hidden_dim']
    projection_dim = config['model']['projection_dim']
    pretrained = config['model']['pretrained']
    
    batch_size = config['linear']['batch_size']
    num_epochs = config['linear']['num_epochs']
    num_output_classes = config['linear']['num_output_classes']
    augment_both = config['linear']['augment_both']
    top_lr = float(config['linear']['top_lr'])
    momentum = float(config['linear']['momentum'])
    weight_decay = float(config['linear']['weight_decay'])
    track_performance = config['linear']['track_performance']
    save_every = int(config['linear']['save_every'])

    # get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get dataset
    _, train_loader, _, test_loader = get_dataset(dataset_name, dataset_path,
                            batch_size=batch_size, 
                            augment_both_views=augment_both,
                            test=True)
    
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
                        )
    else:
        raise NotImplementedError(f"{method_type} not implemented")
    
    if pretrained:
        load_model(ssl_model, config, args.ckpt_path)
    
    ssl_model.to(device)

    # define settings
    Settings = namedtuple('Settings', ['device', 'num_output_classes', 
                                       'top_lr', 'momentum', 'weight_decay', 
                                       'epochs', 'save_every', 
                                       'track_performance'])
    settings = Settings(device=device, num_output_classes=num_output_classes,
                        top_lr=top_lr, momentum=momentum, weight_decay=weight_decay,
                        epochs=num_epochs, save_every=save_every, 
                        track_performance=track_performance)
    
    # initialize wandb run
    wandb.init(project='simclr', 
               config = {
                "experiment_name": experiment_name,
                "dataset_name": dataset_name,
                "batch_size": batch_size,
                "encoder_type": encoder_type,
                "width_multiplier": width_multiplier,
                "hidden_dim": hidden_dim,
                "projection_dim": projection_dim,

            })

    # train linear probing
    embedding_performance(ssl_model, settings, train_loader, test_loader)