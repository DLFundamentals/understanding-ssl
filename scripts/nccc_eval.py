import torch
torch.set_default_dtype(torch.float32)

import torchvision.models as models
import sys, os, argparse, yaml, pandas as pd

# Append the parent directory for utility modules.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions and models.
from utils.dataset_loader import get_dataset
from utils.metrics import KNN, NCCCEval
from utils.eval_utils import load_snapshot, run_few_shot_error_analysis
from models.simclr import SimCLR

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures determinism

set_seed(42)

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

    batch_size = config['training']['batch_size']
    augment_both = config['training']['augment_both']
    

    encoder_type = config['model']['encoder_type']
    width_multiplier = config['model']['width_multiplier']
    projection_dim = config['model']['projection_dim']
    hidden_dim = config['model']['hidden_dim']

    temperature = config['loss']['temperature']

    K = config['evaluation']['K']
    checkpoints_dir = config['evaluation']['checkpoints_dir']
    perform_knn = config['evaluation']['perform_knn']
    perform_cdnv = config['evaluation']['perform_cdnv']
    perform_nccc = config['evaluation']['perform_nccc']
    perform_tsne = config['evaluation']['perform_tsne']

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get dataset
    augment_both = False # override for evaluation
    _, train_loader, _, test_loader, train_labels, _ = get_dataset(dataset_name=dataset_name, 
                                                                    dataset_path=dataset_path,
                                                                    augment_both_views=augment_both,
                                                                    batch_size=batch_size, test=True)
    
    # load model
    if encoder_type == 'resnet50':
        encoder = models.resnet50(pretrained=False)
    else:
        raise NotImplementedError(f"{encoder_type} not implemented")
    
    if method_type == 'simclr':
        ssl_model = SimCLR(model=encoder,
                            dataset=dataset_name,
                            width_multiplier=width_multiplier,
                            hidden_dim=hidden_dim,
                            projection_dim=projection_dim,
                            track_performance=True,
                            K=K)
    else:
        raise NotImplementedError(f"{method_type} not implemented")
    
    # load model checkpoint
    checkpoints_dir = args.ckpt_path
    print(f"Loading checkpoints from {checkpoints_dir}")

    # ===================== NCCC Evaluation =============================== #
    # we do this mainly for few-shot error analysis
    print('Performing NCCC Evaluation')
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
        encoder = models.resnet50(pretrained=False)
        nscl_model = SimCLR(model=encoder,
                            dataset=dataset_name,
                            width_multiplier=width_multiplier,
                            hidden_dim=hidden_dim,
                            projection_dim=projection_dim,
                            track_performance=True,
                            K=K)
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
    ncc_evaluator = NCCCEval(train_loader,
                            train_labels,
                            output_classes=num_output_classes,
                            device=device)

    # few shot error analysis
    output_dir = os.path.join('/home/luthra/understanding-ssl/', args.output_path)
    output_csv_path = os.path.join(output_dir, 'few_shot_analysis.csv')
    if os.path.exists(output_csv_path):
        print('Loading logs from: ', output_csv_path)

    run_few_shot_error_analysis(ncc_evaluator, 
                                ssl_model, 
                                nscl_model, 
                                train_loader, test_loader,
                                output_csv_path, 
                                n_samples_list=[1, 5, 10, 20, 50, 100, 200, 500],
                                repeat=5,
                                )