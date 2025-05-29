import torch
torch.set_default_dtype(torch.float32)

import torchvision.models as models
import sys, os, argparse, yaml, pandas as pd
from glob import glob
from tqdm import tqdm
import numpy as np

# Append the parent directory for utility modules.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions and models.
from utils.dataset_loader import get_dataset
from utils.metrics import cal_cdnv, RepresentationEvaluator
from utils.eval_utils import load_snapshot
from models.simclr import SimCLR

from collections import defaultdict, namedtuple

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures determinism

set_seed(42)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def get_all_cdnvs(evaluator, loader):
    features, labels = evaluator.extract_features(loader)
    cdnv_0 = evaluator.compute_cdnv(torch.from_numpy(features[0]).to(device),
                                        torch.from_numpy(labels).to(device))
    cdnv_1 = evaluator.compute_cdnv(torch.from_numpy(features[1]).to(device),
                                        torch.from_numpy(labels).to(device))
    
    dir_cdnv_0 = evaluator.compute_directional_cdnv(torch.from_numpy(features[0]).to(device),
                                                    torch.from_numpy(labels).to(device))
    dir_cdnv_1 = evaluator.compute_directional_cdnv(torch.from_numpy(features[1]).to(device),
                                                    torch.from_numpy(labels).to(device))
    
    return cdnv_0, cdnv_1, dir_cdnv_0, dir_cdnv_1

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
    perform_cdnv = config['evaluation']['perform_cdnv']

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get dataset
    augment_both = False # override for evaluation
    _, train_loader, _, test_loader, train_labels, test_labels = get_dataset(dataset_name=dataset_name, 
                                                                            dataset_path=dataset_path,
                                                                            augment_both_views=augment_both,
                                                                            batch_size=batch_size, test=True)
    
    # load model
    if encoder_type == 'resnet50':
        encoder1 = models.resnet50(pretrained=False)
        encoder2 = models.resnet50(pretrained=False)
    else:
        raise NotImplementedError(f"{encoder_type} not implemented")
    
    if method_type == 'simclr':
        dcl_model = SimCLR(model=encoder1,
                            dataset=dataset_name,
                            width_multiplier=width_multiplier,
                            hidden_dim=hidden_dim,
                            projection_dim=projection_dim,
                            track_performance=True,)
        
        nscl_model = SimCLR(model=encoder2,
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
    # ===================== CDNV Evaluation =============================== #
    # define settings
    Settings = namedtuple('Settings', ['device', 'num_output_classes'])
    settings = Settings(device=device, num_output_classes=num_output_classes)

    if perform_cdnv:
        print('Performing CDNV Evaluation')
        # load SSL model
        checkpoint_files = os.listdir(checkpoints_dir)
        sorted_checkpoints = sorted(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # remove checkpoints for epochs [10,20, ... till 90]
        # sorted_checkpoints = [ckpt for ckpt in sorted_checkpoints if int(ckpt.split('_')[-1].split('.')[0])]
        
        nscl_checkpoints_dir = args.ckpt_path_nscl
        nscl_checkpoint_files = os.listdir(nscl_checkpoints_dir)
        nscl_sorted_checkpoints = sorted(nscl_checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        # CDNV analysis
        output_dir = os.path.join('/home/luthra/understanding-ssl/', args.output_path)
        output_csv_path = os.path.join(output_dir, 'cdnv_analysis.csv')
        if os.path.exists(output_csv_path):
            print('Loading logs from: ', output_csv_path)
            cdnv_df = pd.read_csv(output_csv_path)
        else:
            # create output dataframe
            cdnv_df = pd.DataFrame(columns=[
                'epoch', 
                'DCL CDNV Train 0', 'DCL CDNV Train 1', 'DCL CDNV Test 0', 'DCL CDNV Test 1',
                'DCL Dir CDNV Train 0', 'DCL Dir CDNV Train 1', 'DCL Dir CDNV Test 0', 'DCL Dir CDNV Test 1',
                'NSCL CDNV Train 0', 'NSCL CDNV Train 1', 'NSCL CDNV Test 0', 'NSCL CDNV Test 1',
                'NSCL Dir CDNV Train 0', 'NSCL Dir CDNV Train 1', 'NSCL Dir CDNV Test 0', 'NSCL Dir CDNV Test 1',
            ])

        for checkpoints in zip(sorted_checkpoints, nscl_sorted_checkpoints):
            epoch = checkpoints[0].split('_')[-1].split('.')[0]
            if epoch in cdnv_df['epoch'].values:
                continue
            # load DCL model
            dcl_checkpoint = checkpoints[0]
            snapshot_path = os.path.join(checkpoints_dir, dcl_checkpoint)
            dcl_model = load_snapshot(snapshot_path, dcl_model, device)
            dcl_model.eval()
            # avoid accidental gradient calculations
            freeze_model(dcl_model)
            print(f'Loading DCL model from {snapshot_path}')
            evaluator1 = RepresentationEvaluator(dcl_model, device, num_output_classes)
            dcl_cdnv_train_0, dcl_cdnv_train_1, dcl_dir_cdnv_train_0, dcl_dir_cdnv_train_1 = get_all_cdnvs(evaluator=evaluator1,
                                                                                                            loader=train_loader)
            dcl_cdnv_test_0, dcl_cdnv_test_1, dcl_dir_cdnv_test_0, dcl_dir_cdnv_test_1 = get_all_cdnvs(evaluator=evaluator1,
                                                                                                            loader=test_loader)


            # load NSCL model
            nscl_checkpoint = checkpoints[1]
            nscl_snapshot_path = os.path.join(nscl_checkpoints_dir, nscl_checkpoint)
            nscl_model = load_snapshot(nscl_snapshot_path, nscl_model, device)
            nscl_model.eval()
            # avoid accidental gradient calculations
            freeze_model(nscl_model)
            print(f'Loading NSCL model from {nscl_snapshot_path}')
            evaluator2 = RepresentationEvaluator(nscl_model, device, num_output_classes)
            nscl_cdnv_train_0, nscl_cdnv_train_1, nscl_dir_cdnv_train_0, nscl_dir_cdnv_train_1 = get_all_cdnvs(evaluator=evaluator2,
                                                                                                            loader=train_loader)
            nscl_cdnv_test_0, nscl_cdnv_test_1, nscl_dir_cdnv_test_0, nscl_dir_cdnv_test_1 = get_all_cdnvs(evaluator=evaluator2,
                                                                                                            loader=test_loader)


            print('DCL CDNV Train 1', dcl_cdnv_train_1)
            print('DCL Dir CDNV Train 1', dcl_dir_cdnv_train_1)
            print('NSCL CDNV Train 1', nscl_cdnv_train_1)
            print('NSCL Dir CDNV Train 1', nscl_dir_cdnv_train_1)

            # clean up
            torch.cuda.empty_cache()

            # create a new row for the dataframe
            new_row = {
                'epoch': epoch,
                'DCL CDNV Train 0': dcl_cdnv_train_0,
                'DCL CDNV Train 1': dcl_cdnv_train_1,
                'DCL CDNV Test 0': dcl_cdnv_test_0,
                'DCL CDNV Test 1': dcl_cdnv_test_1,

                'DCL Dir CDNV Train 0': dcl_dir_cdnv_train_0,
                'DCL Dir CDNV Train 1': dcl_dir_cdnv_train_1,
                'DCL Dir CDNV Test 0': dcl_dir_cdnv_test_0,
                'DCL Dir CDNV Test 1': dcl_dir_cdnv_test_1,

                'NSCL CDNV Train 0': nscl_cdnv_train_0,
                'NSCL CDNV Train 1': nscl_cdnv_train_1,
                'NSCL CDNV Test 0': nscl_cdnv_test_0,
                'NSCL CDNV Test 1': nscl_cdnv_test_1,

                'NSCL Dir CDNV Train 0': nscl_dir_cdnv_train_0,
                'NSCL Dir CDNV Train 1': nscl_dir_cdnv_train_1,
                'NSCL Dir CDNV Test 0': nscl_dir_cdnv_test_0,
                'NSCL Dir CDNV Test 1': nscl_dir_cdnv_test_1,
            }
        
            cdnv_df = pd.concat([cdnv_df, pd.DataFrame([new_row])], ignore_index=True)

        cdnv_df.to_csv(output_csv_path, index=False)