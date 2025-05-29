import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset
import torchvision

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# utils
from utils.dataset_loader import get_dataset
from utils.eval_utils import load_snapshot
from utils.metrics import LinearProbeEval

# model
from models.simclr import SimCLR

import argparse
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import namedtuple
import wandb

# set seed
torch.manual_seed(123)
torch.cuda.manual_seed(123)

def load_model(ssl_model, config, ckpt_path):
    if ckpt_path is None:
        ckpt_path = config['model']['ckpt_path']

    # raise error if still None
    if ckpt_path is None:
        raise ValueError("ckpt_path not provided")
    
    snapshot = torch.load(ckpt_path, map_location='cuda', weights_only=True)
    ssl_model.load_state_dict(snapshot['MODEL_STATE'])
    print(f"Model loaded from {ckpt_path}")


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='SimCLR Linear Probing')
    parser.add_argument('--config', '-c', required=True, 
                        help='path to yaml config file')
    parser.add_argument('--ckpt_path', '-ckpt', default=None,
                        help='path to model checkpoint')
    parser.add_argument('--output_path', '-o', required=True,
                        help='path to save output logs')
    parser.add_argument('--N', '-n', default=1, type=int,
                        help='number of samples for few-shot learning')
    parser.add_argument('--seed', '-s', default=1, type=int,)                   
    args = parser.parse_args()

    # load config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # load config parameters
    experiment_name = config['experiment_name']
    method_type = config['method_type']
    supervision = config['supervision']

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
    train_dataset, train_loader, test_dataset, test_loader, train_labels, test_labels = get_dataset(dataset_name, dataset_path,
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
    
    checkpoints_dir = args.ckpt_path
    checkpoint_files = os.listdir(checkpoints_dir)
    sorted_checkpoints = sorted(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    best_checkpoint = sorted_checkpoints[-1]
    snapshot_path = os.path.join(checkpoints_dir, best_checkpoint)
    ssl_model = load_snapshot(snapshot_path, ssl_model, device)
    ssl_model.eval()
    ssl_model.to(device)
    print(f'Loading DCL model from {snapshot_path}')

    # avoid accidental gradient calculation
    for param in ssl_model.parameters():
        param.requires_grad = False

    # define settings
    Settings = namedtuple('Settings', ['device', 'num_output_classes', 
                                       'top_lr', 'momentum', 'weight_decay', 
                                       'epochs', 'save_every', 
                                       'track_performance'])
    settings = Settings(device=device, num_output_classes=num_output_classes,
                        top_lr=top_lr, momentum=momentum, weight_decay=weight_decay,
                        epochs=num_epochs, save_every=save_every, 
                        track_performance=track_performance)
    
    # initialize linear probe evaluator
    linear_evaluator = LinearProbeEval(
        ssl_model,
        train_loader,
        num_output_classes,
        num_epochs,
        top_lr,
        device,
        labels=None,
        log_every=100,
        log_to_wandb=False,
        wandb_project="linear-prob-eval",
        wandb_name="full-shot",
        train_labels=train_labels,
        test_labels=test_labels,
    )

    # check output directory and logs file
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        print(f"Created output directory: {output_path}")
    output_logs_file = os.path.join(output_path, f'few_shot_lin_prob_{supervision}_new.csv')
    train_acc = []
    test_acc = []

    if os.path.exists(output_logs_file):
        few_shot_df = pd.read_csv(output_logs_file)
    else:
        few_shot_df = pd.DataFrame(columns=[
            'Number of Shots', 'Train Acc', 'Test Acc'
        ])
        print('Created a new dataframe for logging results.')

    # set number of samples for few-shot learning
    # N = [1, 5, 10, 20, 50, 100, 200, 500]
    N = [args.N] # can use this for CLI arguments
    for n_samples in N:
        if n_samples in few_shot_df['Number of Shots'].values:
            print(f"Evaluation exists for {n_samples} samples!")
            continue
        wandb_name = f'few-shot-{n_samples}'
        res, res_test = linear_evaluator.evaluate(test_loader, 
                                        n_samples=n_samples,
                                        repeat=5,
                                        embedding_layer=[1],
                                        wandb_name=wandb_name)
        
        train_acc.append(res)
        test_acc.append(res_test)

        new_row = {
            'Number of Shots': n_samples,
            'Train Acc': res,
            'Test Acc': res_test
        }

        few_shot_df = pd.concat([few_shot_df, pd.DataFrame([new_row])], ignore_index=True)

    few_shot_df.to_csv(output_logs_file, index=False)