import torch
torch.set_default_dtype(torch.float32)

import torchvision.models as models


import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# utils
from utils.augmentations import get_transforms
from utils.dataset_loader import get_dataset
from utils.losses import NTXentLoss
from utils.metrics import KNN, NCCCEval

# model
from models.simclr import SimCLR

# visualization
from scripts.visualize import plot_tsne, line_plot

import argparse
import yaml
from glob import glob
from tqdm import tqdm
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
    parser.add_argument('--ckpt_path', '-ckpt', required=True, help='path to model checkpoint')
    args = parser.parse_args()

    # load config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # load config parameters required for evaluation
    experiment_name = config['experiment_name']
    method_type = config['method_type']

    dataset_name = config['dataset']['name']
    dataset_path = config['dataset']['path']

    batch_size = config['training']['batch_size']
    augment_both = config['training']['augment_both']
    K = config['evaluation']['K']
    checkpoints_dir = config['evaluation']['checkpoints_dir']
    perform_knn = config['evaluation']['perform_knn']
    perform_cdnv = config['evaluation']['perform_cdnv']
    perform_nccc = config['evaluation']['perform_nccc']
    perform_tsne = config['evaluation']['perform_tsne']

    encoder_type = config['model']['encoder_type']
    width_multiplier = config['model']['width_multiplier']
    projection_dim = config['model']['projection_dim']
    hidden_dim = config['model']['hidden_dim']



    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get dataset
    batch_size = 128
    augment_both = False
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
    knn_accs_train = defaultdict(list)
    knn_accs_test = defaultdict(list)
    x_axis = [] # for line plots

    print(f"Loading checkpoints from {checkpoints_dir}")
    # all_checkpoints = os.listdir(checkpoints_dir)
    # # sort them by epoch number
    # all_checkpoints = sorted(all_checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # for checkpoint_path in tqdm(all_checkpoints):

    snapshot_path = "/home/luthra/understanding-ssl/experiments/simclr/cifar10/snapshot.pth"
    snapshot = torch.load(snapshot_path, map_location=device, weights_only=True)
    state_dict = snapshot['MODEL_STATE']
    epochs_trained = snapshot['EPOCHS_RUN']
    print(f"Loaded model from epoch {epochs_trained}")
    ssl_model.load_state_dict(state_dict)
    ssl_model.to(device)
    ssl_model.eval()
    print("Model loaded successfully")

    # evaluators
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

    if perform_knn:
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
                
    if perform_nccc:
        ncc_evaluator = NCCCEval(train_loader,
                                 output_classes=10,
                                 device=device)
        
        avg_nccc_score = ncc_evaluator.evaluate(ssl_model, train_loader,
                               n_samples=None, repeat=1)
        print("====== 🚆 Training Dataset =========")
        print(f"Average NCCC Score for embedding layer 0: {avg_nccc_score[0]}")
        print(f"Average NCCC Score for embedding layer 1: {avg_nccc_score[1]}")

        # test dataset
        avg_nccc_score = ncc_evaluator.evaluate(ssl_model, test_loader,
                               n_samples=None, repeat=1)
        print("====== 🚆 Test Dataset =========")
        print(f"Average NCCC Score for embedding layer 0: {avg_nccc_score[0]}")
        print(f"Average NCCC Score for embedding layer 1: {avg_nccc_score[1]}")