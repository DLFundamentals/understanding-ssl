import torch
torch.set_default_dtype(torch.float32)

import torchvision.models as models
import sys, os, argparse, yaml, pandas as pd
from glob import glob
from tqdm import tqdm

# Append the parent directory for utility modules.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions and models.
from utils.augmentations import get_transforms
from utils.dataset_loader import get_dataset
from utils.losses import NTXentLoss, WeakNTXentLoss
from utils.metrics import KNN, NCCCEval, anisotropy
from utils.analysis import get_ssl_minus_scl_loss
from utils.eval_utils import load_snapshot, evaluate_losses_for_ssl, \
                              run_few_shot_error_analysis
from models.simclr import SimCLR
from scripts.visualize import plot_tsne, line_plot

from collections import defaultdict

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
    parser.add_argument('--perform_loss_diff', type=bool, help="calculates all the losses and differences")
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
    _, train_loader, _, test_loader = get_dataset(dataset_name=dataset_name, 
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
    # breakpoint()
    if args.perform_loss_diff:
        temperature = 1.0 # override for evaluation
        output_logs = os.path.join('/home/luthra/understanding-ssl/', args.output_path)
        output_logs_file = os.path.join(output_logs, 'losses.csv')
        # example: /home/luthra/understanding-ssl/logs/cifar10/simclr/exp1/output.csv
        output_df = evaluate_losses_for_ssl(ssl_model, 
                                            checkpoints_dir, 
                                            train_loader, test_loader,
                                            temperature, 
                                            device, 
                                            output_logs_file)
    

    # ===================== NCCC Evaluation =============================== #
    # we do this mainly for few-shot error analysis
    if perform_nccc:
        # load SSL model
        checkpoint_files = os.listdir(checkpoints_dir)
        sorted_checkpoints = sorted(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        best_checkpoint = sorted_checkpoints[-1]
        snapshot_path = os.path.join(checkpoints_dir, best_checkpoint)
        ssl_model = load_snapshot(snapshot_path, ssl_model, device)

        # define and load NSCL model
        if method_type == 'simclr':
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
        checkpoint_files = os.listdir(nscl_checkpoints_dir)
        sorted_checkpoints = sorted(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        best_checkpoint = sorted_checkpoints[-1]
        snapshot_path = os.path.join(checkpoints_dir, best_checkpoint)
        nscl_model = load_snapshot(snapshot_path, nscl_model, device)


        ncc_evaluator = NCCCEval(train_loader,
                                 output_classes=num_output_classes,
                                 device=device)
        
        # few shot error analysis
        output_csv_path = os.path.join('/home/luthra/understanding-ssl/experiments', args.output_path)
        run_few_shot_error_analysis(ncc_evaluator, 
                                    ssl_model, 
                                    nscl_model, 
                                    train_loader, test_loader,
                                    output_csv_path, 
                                    n_samples_list=[1, 5, 10, 20, 50, 100, 200], 
                                    repeat=5)
        

        
     # ===================== KNN Evaluation =============================== #
    knn_accs_train = defaultdict(list)
    knn_accs_test = defaultdict(list)
    x_axis = [] # for line plots
    knn_evaluator = KNN(ssl_model, K, device=device)
    
    if perform_knn:
        train_acc, test_acc = knn_evaluator.knn_eval(train_loader)

        if not isinstance(train_acc, list):
            train_acc = [train_acc]
        if not isinstance(K, list):
            K = [K]
        if test_acc is not None:
            if not isinstance(test_acc, list):
                test_acc = [test_acc]
        
        for i, k in enumerate(K):
            knn_accs_train[k].append(train_acc[i])
            if test_acc:
                knn_accs_test[k].append(test_acc[i])

        # plot training performance
        for k in K:
            line_plot(x_axis, knn_accs_train[k], x_label='Epochs', y_label='KNN Accuracy',
                    title=f"KNN Train Accuracy for K={k}",
                    output_dir="/home/luthra/understanding-ssl/experiments/", 
                    experiment_name=f"{experiment_name}/",
                    filename=f"knn_train_acc_k{k}.png")

            if test_acc:
                line_plot(x_axis, knn_accs_test[k], x_label='Epochs', y_label='KNN Accuracy',
                        title=f"KNN Test Accuracy for K={k}",
                        output_dir="/home/luthra/understanding-ssl/experiments/", 
                    experiment_name=f"{experiment_name}/",
                        filename=f"knn_test_acc_k{k}.png")

    if perform_tsne:
    # plot tSNE
        features, labels = knn_evaluator.extract_features(train_loader)
        # random_indices = torch.randperm(features.size(0))[:5000]
        # num_classes = len(torch.unique(labels[random_indices]))
        # pick indices such that all classes are present approximately equally
        indices = []
        for i in range(10):
            indices.append(torch.where(labels == i)[0][:250])
        indices = torch.cat(indices)
        
        plot_tsne(features[indices], labels[indices], num_classes=10,
                output_dir="/home/luthra/understanding-ssl/experiments/", 
                experiment_name=f"{experiment_name}/",
                filename=f"tsne_epoch.png") #TODO
