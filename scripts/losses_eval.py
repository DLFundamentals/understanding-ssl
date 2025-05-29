import torch
torch.set_default_dtype(torch.float32)

import torchvision.models as models
import sys, os, argparse, yaml, pandas as pd

# Append the parent directory for utility modules.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions and models.
from utils.dataset_loader import get_dataset
from utils.eval_utils import evaluate_losses_for_ssl
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
    _, train_loader, _, test_loader, _, _ = get_dataset(dataset_name=dataset_name, 
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
    # ===================== Theorem 1 =============================== #
    temperature = 1.0 # override for evaluation
    batch_size = 1024 # override for evaluation
    output_dir= args.output_path
    os.makedirs(output_dir, exist_ok=True)
    output_logs_file = os.path.join(output_dir, 'losses.csv')
    # example: /home/luthra/understanding-ssl/logs/cifar10/simclr/exp1/output.csv
    output_df = evaluate_losses_for_ssl(ssl_model, 
                                        checkpoints_dir, 
                                        train_loader, test_loader,
                                        temperature, 
                                        device, 
                                        output_logs_file)
