import warnings
warnings.filterwarnings("ignore", 
    message=".*force_all_finite.*renamed to 'ensure_all_finite'.*")

from collections import defaultdict
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torchvision

import sys, os, yaml, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.simclr import SimCLR
from utils.dataset_loader import get_dataset
from utils.eval_utils import load_snapshot, get_fewshot_loader

# umap imports
from umap import UMAP
from sklearn.preprocessing import StandardScaler

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures determinism

set_seed(42)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def plot_umap_vis(features, labels, 
                  objective='dcl',
                  dataset_name='imagenet',
                  output_path=None,
                  epoch=0):
    # normalize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # fit umap
    umap_model = UMAP(n_components=2, random_state=123)
    proj_2d = umap_model.fit_transform(features_scaled)

    # plotting
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    colors_dict = defaultdict()
    for c in classes:
        colors_dict[c] = colors.pop(0)

    colors_map = [colors_dict[label.item()] for label in labels]
    fig, ax = plt.subplots(figsize=(4, 4))
    scatter = ax.scatter(proj_2d[:, 0], proj_2d[:, 1], c=colors_map, s=6)

    # create legends
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"{output_path}/umap_{dataset_name}_{objective}_epoch{epoch}.pdf", bbox_inches='tight')
    plt.show()

@torch.no_grad
def extract_inputs_for_umap(model, umap_loader):

    all_encoder_features = []
    all_proj_features = []
    all_labels = []

    with torch.no_grad():
        for batch in umap_loader:
            _, image, label = batch
            image = image.to('cuda')

            encoder_features, proj_features = model(image)

            all_encoder_features.append(encoder_features.detach().cpu())
            all_proj_features.append(proj_features.detach().cpu())
            all_labels.append(label.detach().cpu())

    # stack everything
    all_encoder_features = torch.cat(all_encoder_features, dim=0)
    all_proj_features = torch.cat(all_proj_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_encoder_features, all_proj_features, all_labels

# =============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="General Evluation Script")
    parser.add_argument('--config', '-c', required=True, help='path to yaml config file')
    parser.add_argument('--ckpt_path', '-ckpt', required=True, default=None,
                        help='path to model checkpoint')
    parser.add_argument('--ckpt_path_nscl', '-ckpt_nscl', default=None,
                        help='path to model checkpoint trained with NSCL')
    parser.add_argument('--output_path', '-out', required=True, default=None,
                        help='path to save logs')
    args = parser.parse_args()

    # load config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # load config parameters required for evaluation
    experiment_name = config['experiment_name']
    method_type = config['method_type']

    dataset_name = config['dataset']['name']
    dataset_path = config['dataset']['path']
    num_output_classes = config['dataset']['num_output_classes']

    encoder_type = config['model']['encoder_type']
    width_multiplier = config['model']['width_multiplier']
    projection_dim = config['model']['projection_dim']
    hidden_dim = config['model']['hidden_dim']

    temperature = config['loss']['temperature']

    batch_size = config['evaluation']['batch_size']

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get dataset
    augment_both = False # override for evaluation
    train_dataset, train_loader, test_dataset, test_loader, train_labels, _ = get_dataset(dataset_name=dataset_name, 
                                            dataset_path=dataset_path,
                                            augment_both_views=augment_both,
                                            batch_size=batch_size, test=True)
    
    # load model
    if encoder_type == 'resnet50':
        encoder1 = torchvision.models.resnet50(pretrained=False)
        encoder2 = torchvision.models.resnet50(pretrained=False)
        encoder3 = torchvision.models.resnet50(pretrained=False)
    else:
        raise NotImplementedError(f"{encoder_type} not implemented")
    
    if method_type == 'simclr':
        random_model = SimCLR(model=encoder1,
                            dataset=dataset_name,
                            width_multiplier=width_multiplier,
                            hidden_dim=hidden_dim,
                            projection_dim=projection_dim,
                            track_performance=True,)
        random_model.eval()
        freeze_model(random_model)
        random_model.to(device)

        dcl_model = SimCLR(model=encoder2,
                            dataset=dataset_name,
                            width_multiplier=width_multiplier,
                            hidden_dim=hidden_dim,
                            projection_dim=projection_dim,
                            track_performance=True,)
        
        nscl_model = SimCLR(model=encoder3,
                            dataset=dataset_name,
                            width_multiplier=width_multiplier,
                            hidden_dim=hidden_dim,
                            projection_dim=projection_dim,
                            track_performance=True)
    else:
        raise NotImplementedError(f"{method_type} not implemented")
    
    # load model checkpoint
    checkpoints_dir = args.ckpt_path
    print(f"Loading checkpoints from {checkpoints_dir}")

    # ===================== UMAP Visualization =============================== #
    print('Starting UMAP Visualization 👁️👁️')

    # load SSL model checkpoints
    checkpoint_files = os.listdir(checkpoints_dir)
    sorted_checkpoints = sorted(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # store specific checkpoints 
    sorted_checkpoints = [ckpt for ckpt in sorted_checkpoints if int(ckpt.split('_')[-1].split('.')[0]) in [10, 100, 400, 500, 700, 1000, 1500, 1900]]
    
    nscl_checkpoints_dir = args.ckpt_path_nscl
    nscl_checkpoint_files = os.listdir(nscl_checkpoints_dir)
    nscl_sorted_checkpoints = sorted(nscl_checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    nscl_sorted_checkpoints = [ckpt for ckpt in nscl_sorted_checkpoints if int(ckpt.split('_')[1].split('.')[0])  in [10, 100, 400, 500, 700, 1000, 1500, 1900]]

    # get path to save outputs
    output_dir = os.path.join('/home/luthra/understanding-ssl/', args.output_path)
    os.makedirs(output_dir, exist_ok=True)

    
    # run the whole thing for 5 randomly sampled classes
    for run in range(10):
        print("="*20)
        print(f"Going for run no. {run} 🏃‍♂️‍➡️🏃‍♂️‍➡️🏃‍♂️‍➡️")
        output_umap_dir = f"{output_dir}/run_{run}"
        os.makedirs(output_umap_dir, exist_ok=True)

        classes = np.random.choice(num_output_classes, 5, replace=False)
        umap_loader = get_fewshot_loader(n_samples=200, dataset=train_dataset, labels=train_labels,
                                batch_size=batch_size, classes=classes)
        
        # random initialization
        print("Evaluating random model 🧮")
        encoder_features, proj_features, labels = extract_inputs_for_umap(random_model, umap_loader)
        plot_umap_vis(proj_features, labels,
                    objective='random', 
                    dataset_name=dataset_name,
                    output_path=output_umap_dir,
                    epoch=0)

        for checkpoints in zip(sorted_checkpoints, nscl_sorted_checkpoints):
            epoch = checkpoints[0].split('_')[-1].split('.')[0]
            epoch_nscl = checkpoints[1].split('_')[-1].split('.')[0]

            # load DCL model
            dcl_checkpoint = checkpoints[0]
            snapshot_path = os.path.join(checkpoints_dir, dcl_checkpoint)
            dcl_model = load_snapshot(snapshot_path, dcl_model, device)
            dcl_model.eval()
            # avoid accidental gradient calculations
            freeze_model(dcl_model)
            dcl_model.to(device)
            print(f'Loading DCL model from {snapshot_path}')

            # load NSCL model
            nscl_checkpoint = checkpoints[1]
            nscl_snapshot_path = os.path.join(nscl_checkpoints_dir, nscl_checkpoint)
            nscl_model = load_snapshot(nscl_snapshot_path, nscl_model, device)
            nscl_model.eval()
            # avoid accidental gradient calculations
            freeze_model(nscl_model)
            nscl_model.to(device)
            print(f'Loading NSCL model from {nscl_snapshot_path}')
            

            # dcl
            print("Evaluating DCL model 🧮")
            encoder_features, proj_features, labels = extract_inputs_for_umap(dcl_model, umap_loader)
            plot_umap_vis(proj_features, labels,
                        objective='dcl', 
                        dataset_name=dataset_name,
                        output_path=output_umap_dir,
                        epoch=epoch)

            # nscl
            print("Evaluating NSCL model 🧮")
            encoder_features, proj_features, labels = extract_inputs_for_umap(nscl_model, umap_loader)
            plot_umap_vis(proj_features, labels,
                        objective='nscl', 
                        dataset_name=dataset_name,
                        output_path=output_umap_dir,
                        epoch=epoch_nscl)


        # clean up
        torch.cuda.empty_cache()