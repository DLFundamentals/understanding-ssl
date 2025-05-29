import torch
import torchvision

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# utils
from utils.dataset_loader import get_dataset
from utils.analysis import embedding_performance
from utils.eval_utils import load_snapshot
from utils.metrics import LinearProbeEval

# model
from models.simclr import SimCLR

import argparse
import yaml
import random
import pandas as pd
import numpy as np
from collections import namedtuple
import wandb

# set seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures determinism
    random.seed(seed)

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
    parser.add_argument('--ckpt_path', '-ckpt', required=True, default=None,
                        help='path to model checkpoint')
    parser.add_argument('--ckpt_path_nscl', '-ckpt_nscl', default=None,
                        help='path to model checkpoint trained with NSCL')
    parser.add_argument('--output_path', '-out', required=True, default=None,
                        help='path to save logs')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

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

    # load model checkpoint
    checkpoints_dir = args.ckpt_path
    print(f"Loading checkpoints from {checkpoints_dir}")

    # load model
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
                            track_performance=True,)
    else:
        raise NotImplementedError(f"{method_type} not implemented")

    # load SSL model
    checkpoint_files = os.listdir(checkpoints_dir)
    sorted_checkpoints = sorted(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    best_checkpoint = sorted_checkpoints[-1]
    snapshot_path = os.path.join(checkpoints_dir, best_checkpoint)
    ssl_model = load_snapshot(snapshot_path, ssl_model, device)
    ssl_model.eval()
    print(f'Loading DCL model from {snapshot_path}')

    # define and load NSCL model
    if method_type == 'simclr':
        encoder = torchvision.models.resnet50(pretrained=False)
        nscl_model = SimCLR(model=encoder,
                            dataset=dataset_name,
                            width_multiplier=width_multiplier,
                            hidden_dim=hidden_dim,
                            projection_dim=projection_dim,
                            track_performance=True,)
    else:
        raise NotImplementedError(f"{method_type} not implemented")
    
    nscl_checkpoints_dir = args.ckpt_path_nscl
    nscl_checkpoint_files = os.listdir(nscl_checkpoints_dir)
    nscl_sorted_checkpoints = sorted(nscl_checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    nscl_best_checkpoint = nscl_sorted_checkpoints[-1]
    nscl_snapshot_path = os.path.join(nscl_checkpoints_dir, nscl_best_checkpoint)
    nscl_model = load_snapshot(nscl_snapshot_path, nscl_model, device)
    nscl_model.eval()
    print(f'Loading NSCL model from {nscl_snapshot_path}')

    # avoid accidental memory build-up
    for param in ssl_model.parameters():
        param.requires_grad = False
    for param in nscl_model.parameters():
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


    output_path = args.output_path
    output_logs_file = os.path.join(output_path, f'few_shot_lin_prob_{args.seed}.csv')
    N = [1, 5, 10, 20, 50, 100, 200, 300, 500, 1000]

    if os.path.exists(output_logs_file):
        few_shot_df = pd.read_csv(output_logs_file)
    else:
        few_shot_df = pd.DataFrame(columns=[
            'Number of Shots',
            'DCL Train Acc', 'DCL Test Acc',
            'NSCL Train Acc', 'NSCL Test Acc'
        ])
        print('Created a new dataframe for logging results.')

    print("="*20)
    classes_group = random.sample(range(0, num_output_classes), 2) # pick 2 classes randomly
    print(f'Current random group: {classes_group}')

    # get dataset
    train_dataset, train_loader, test_dataset, test_loader, train_labels, test_labels = get_dataset(dataset_name, dataset_path,
                                                                                                    batch_size=batch_size, 
                                                                                                    augment_both_views=augment_both,
                                                                                                    test=True,
                                                                                                    classes = classes_group)
    
    # define evaluator instance
    ssl_linear_evaluator = LinearProbeEval(
        ssl_model,
        train_loader,
        output_classes=len(np.unique(train_labels)),
        epochs=501,
        lr=top_lr,
        device=device,
        labels=classes_group,
        log_every=100,
        log_to_wandb=False,
        wandb_project="linear-prob-eval",
        wandb_name="full-shot",
        train_labels=train_labels,
        test_labels=test_labels,
    )

    # define evaluator instance
    nscl_linear_evaluator = LinearProbeEval(
        nscl_model,
        train_loader,
        output_classes=len(np.unique(train_labels)),
        epochs=501,
        lr=top_lr,
        device=device,
        labels=classes_group,
        log_every=100,
        log_to_wandb=False,
        wandb_project="linear-prob-eval",
        wandb_name="full-shot",
        train_labels=train_labels,
        test_labels=test_labels,
    )

    for n_samples in N:
        if n_samples in few_shot_df['Number of Shots'].values:
            print(f"Evaluation exists for {n_samples} samples!")
            continue

        dcl_train_acc = []
        dcl_test_acc = []
        nscl_train_acc = []
        nscl_test_acc = []
        
        # run multiple times or different seeds?
        res, res_test = ssl_linear_evaluator.evaluate(test_loader, 
                                                n_samples=n_samples,
                                                repeat=1,
                                                embedding_layer=[1],
                                                wandb_name=None)
        dcl_train_acc.append(res)
        dcl_test_acc.append(res_test)


        res, res_test = nscl_linear_evaluator.evaluate(test_loader, 
                                                n_samples=n_samples,
                                                repeat=1,
                                                embedding_layer=[1],
                                                wandb_name=None)
        
        nscl_train_acc.append(res)
        nscl_test_acc.append(res_test)

        new_row = {
            'Number of Shots': n_samples,
            'DCL Train Acc': np.mean(dcl_train_acc),
            'DCL Test Acc': np.mean(dcl_test_acc),
            'NSCL Train Acc': np.mean(nscl_train_acc),
            'NSCL Test Acc': np.mean(nscl_test_acc)
        }

        few_shot_df = pd.concat([few_shot_df, pd.DataFrame([new_row])], ignore_index=True)

    few_shot_df.to_csv(output_logs_file, index=False)