import os, sys, random
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset, Subset

# Append the parent directory for utility modules.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.losses import NTXentLoss, WeakNTXentLoss, ContrastiveLoss
from utils.metrics import KNN, NCCCEval, anisotropy
from utils.dataset_loader import get_dataset
# from utils.analysis import get_ssl_minus_scl_loss

def load_snapshot(snapshot_path, model, device):
    snapshot = torch.load(snapshot_path, map_location=device, weights_only=True)
    state_dict = snapshot['MODEL_STATE']
    epochs_trained = snapshot['EPOCHS_RUN']
    print(f"Loaded model from epoch {epochs_trained}")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("SSL Model loaded successfully")
    return model

def get_label_map(labels):
    """
    map the labels to the output_classes
    """
    label_map = {}
    for i, label in enumerate(labels):
        label_map[label] = i

    return label_map

def get_ssl_minus_scl_loss(ssl_model, loader, ssl_criterion, weak_scl_criterion,
                           cl_criterion=None,
                           labels_for_mapping = None, 
                           device='cuda'): 
    if labels_for_mapping:
        label_map = get_label_map(labels_for_mapping)
    ssl_model.eval()
    with torch.no_grad():
        total_ssl_loss = 0.0
        total_scl_loss = 0.0
        if cl_criterion:
            total_cl_loss = 0.0
        for batch in tqdm(loader):
            view1, view2, labels = batch
            view1 = view1.to(device)
            view2 = view2.to(device)
            labels = labels.to(device)
            if labels_for_mapping:
                labels = torch.tensor([label_map[i.item()] for i in labels],
                                      device=device)

            # forward pass
            view1_features, view1_proj = ssl_model(view1)
            view2_features, view2_proj = ssl_model(view2)

            # calculate ssl loss
            ssl_loss = ssl_criterion(view1_proj, view2_proj, labels)
            total_ssl_loss += ssl_loss.item()

            # calculate weak scl loss
            weak_scl_loss = weak_scl_criterion(view1_proj, view2_proj, labels)
            total_scl_loss += weak_scl_loss.item()

            if cl_criterion:
                cl_loss = cl_criterion(view1_proj, view2_proj, labels)
                total_cl_loss += cl_loss.item()

        torch.cuda.empty_cache()

        print(f"Total SSL Loss: {total_ssl_loss/len(loader)}")
        print(f"Total Weak SCL Loss: {total_scl_loss/len(loader)}")
        if cl_criterion:
            print(f"Total CL Loss: {total_cl_loss/len(loader)}")

    diff = total_ssl_loss - total_scl_loss
    
    if cl_criterion:
        diff_cl = total_cl_loss - total_scl_loss
        return diff/len(loader), diff_cl/len(loader), total_cl_loss/len(loader), \
        total_ssl_loss/len(loader), total_scl_loss/len(loader)

    return diff/len(loader), total_ssl_loss/len(loader), total_scl_loss/len(loader)


def evaluate_losses_for_ssl(ssl_model, checkpoints_dir, train_loader, test_loader,
                         temperature, device, output_logs_file):
    """
    Evaluates a series of model snapshots by computing loss differences and other metrics.
    
    Loads each checkpoint from `checkpoints_dir`, updates the provided ssl_model,
    computes losses on train and test loaders using NTXentLoss and WeakNTXentLoss criteria,
    and appends the results into a DataFrame which is then saved to CSV.
    
    Parameters:
        ssl_model: The self-supervised model instance (e.g., a SimCLR model).
        checkpoints_dir (str): Directory where checkpoint files are stored.
        train_loader: DataLoader for the training set.
        test_loader: DataLoader for the test set.
        temperature (float): Temperature for the loss functions.
        device (str): Device to run computations on (e.g., 'cuda' or 'cpu').
        output_logs_file (str): Full path to the CSV file in which to log results.
        
    Returns:
        output_df (pd.DataFrame): DataFrame containing evaluation metrics for each checkpoint.
    """
    # Instantiate loss criteria with the given temperature.
    cl_criterion = ContrastiveLoss(temperature, device=device)
    ssl_criterion = NTXentLoss(temperature, device=device)
    weak_scl_criterion = WeakNTXentLoss(temperature, device=device)
    
    # Get checkpoint filenames and sort them based on the epoch number extracted from the filename.
    checkpoint_files = os.listdir(checkpoints_dir)
    sorted_checkpoints = sorted(
        checkpoint_files, 
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    
    # Load existing DataFrame or create a new one with required columns.
    if os.path.exists(output_logs_file):
        output_df = pd.read_csv(output_logs_file)
    else:
        output_df = pd.DataFrame(columns=[
            'epoch', 'cl_loss', 'cl_loss_test', 'dcl_loss', 'dcl_loss_test',
            'nscl_loss', 'nscl_loss_test', 'diff', 'diff_test', 'diff_cl',
            'diff_cl_test'
        ])
        print(f"Created new DataFrame for logging results.")
    
    # Loop through each checkpoint file.
    for checkpoint in sorted_checkpoints:
        print(f"Calculating loss for checkpoint: {checkpoint}")
        try:
            cur_epoch = int(checkpoint.split('_')[-1].split('.')[0])
        except ValueError:
            print(f"Skipping checkpoint {checkpoint} due to epoch parse error.")
            continue
        
        # Skip checkpoint if already logged.
        if cur_epoch in output_df['epoch'].values:
            continue
        
        snapshot_path = os.path.join(checkpoints_dir, checkpoint)
        ssl_model = load_snapshot(snapshot_path, ssl_model, device)
        
        # Compute the losses on the training set.
        diff, diff_cl, cl_loss, dcl_loss, nscl_loss = get_ssl_minus_scl_loss(
            ssl_model, train_loader, ssl_criterion, weak_scl_criterion,
            cl_criterion=cl_criterion
        )
        # Compute the losses on the test set.
        diff_test, diff_cl_test, cl_loss_test, dcl_loss_test, nscl_loss_test = get_ssl_minus_scl_loss(
            ssl_model, test_loader, ssl_criterion, weak_scl_criterion,
            cl_criterion=cl_criterion
        )
        
        # Create a new row of results.
        new_row = {
            'epoch': cur_epoch,
            'cl_loss': cl_loss,
            'cl_loss_test': cl_loss_test,
            'dcl_loss': dcl_loss,
            'dcl_loss_test': dcl_loss_test,
            'nscl_loss': nscl_loss,
            'nscl_loss_test': nscl_loss_test,
            'diff': diff,
            'diff_test': diff_test,
            'diff_cl': diff_cl,
            'diff_cl_test': diff_cl_test,
            
        }
        # Append the new row to the DataFrame.
        output_df = pd.concat([output_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save the DataFrame to CSV.
    output_df.to_csv(output_logs_file, index=False)
    print(f"Output logs saved to: {output_logs_file}")
    
    return output_df


def run_few_shot_error_analysis(ncc_evaluator, ssl_model, nscl_model, train_loader, test_loader,
                                output_csv_path, n_samples_list=[1, 5, 10, 20, 50, 100], repeat=5,
                                **kwargs):
    """
    Performs few-shot error analysis for two models (e.g., a contrastive (CL) model and a non-scaled version (NSCL)).
    
    For each number in n_samples_list, the function evaluates both models on the train and test loaders.
    The evaluation metric of interest is taken from the second element of the returned tuple from ncc_evaluator.evaluate.
    
    Parameters:
        ncc_evaluator: An object that provides an evaluate(model, loader, n_samples, repeat) method.
        ssl_model: The primary self-supervised model to evaluate (e.g., your CL model).
        nscl_model: The secondary model (e.g., a NSCL variant) to evaluate.
        train_loader: DataLoader for the training dataset.
        test_loader: DataLoader for the test dataset.
        output_csv_path (str): Full path where the CSV log should be saved.
        n_samples_list (list): List of integers representing few-shot sample sizes.
        repeat (int): Number of evaluation repetitions per few-shot configuration.
    
    Returns:
        few_shot_df (pd.DataFrame): DataFrame containing few-shot evaluation metrics.
    """

    if os.path.exists(output_csv_path):
        few_shot_df = pd.read_csv(output_csv_path)
    else:
        few_shot_df = pd.DataFrame(columns=[
            "Number of Shots", "CL Train", "CL Test",
            "NSCL Train", "NSCL Test"
        ])
    

    total_eval_runs = 5
    for n_samples in n_samples_list:
        ssl_few_shot_accs_train = []
        ssl_few_shot_accs_test = []
        nscl_few_shot_accs_train = []
        nscl_few_shot_accs_test = []
        print(f"Evaluating for number of samples: {n_samples}")
        if n_samples in few_shot_df['Number of Shots'].values:
            print('Evaluation exists!')
            continue
        
        for _ in range(total_eval_runs):
            # Evaluate the CL (ssl_model) on the training set
            print(f"Evaluating SSL model on training set")
            ssl_accs_train = ncc_evaluator.evaluate(ssl_model, train_loader, n_samples=n_samples, repeat=repeat)
            ssl_few_shot_accs_train.append(ssl_accs_train[1])
            # Evaluate the CL model on the test set
            print(f"Evaluating SSL model on test set")
            ssl_accs_test = ncc_evaluator.evaluate(ssl_model, test_loader, n_samples=n_samples, repeat=repeat)
            ssl_few_shot_accs_test.append(ssl_accs_test[1])
            
            # Evaluate the NSCL (nscl_model) on the training set
            print(f"Evaluating NSCL model on training set")
            nscl_accs_train = ncc_evaluator.evaluate(nscl_model, train_loader, n_samples=n_samples, repeat=repeat)
            nscl_few_shot_accs_train.append(nscl_accs_train[1])
            # Evaluate the NSCL model on the test set
            print(f"Evaluating NSCL model on test set")
            nscl_accs_test = ncc_evaluator.evaluate(nscl_model, test_loader, n_samples=n_samples, repeat=repeat)
            nscl_few_shot_accs_test.append(nscl_accs_test[1])

        new_row = {
            "Number of Shots": n_samples,
            "CL Train": np.mean(ssl_few_shot_accs_train),
            "CL Test": np.mean(ssl_few_shot_accs_test),
            "CL Train STD": np.std(ssl_few_shot_accs_train),
            "CL Test STD": np.std(ssl_few_shot_accs_test),
            "NSCL Train": np.mean(nscl_few_shot_accs_train),
            "NSCL Test": np.mean(nscl_few_shot_accs_test),
            "NSCL Train STD": np.std(nscl_few_shot_accs_train),
            "NSCL Test STD": np.std(nscl_few_shot_accs_test),

        }

        few_shot_df = pd.concat([few_shot_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save the DataFrame to the specified CSV file.
    few_shot_df.to_csv(output_csv_path, index=False)
    print(f"Few-shot analysis saved to: {output_csv_path}")
    
    return few_shot_df


def run_few_shot_error_analysis_per_C(ncc_evaluator, 
                                ssl_model, nscl_model, 
                                train_loader, test_loader,
                                n_samples=10, repeat=5):
    """
    """
    
    ssl_accs_train = ncc_evaluator.evaluate(ssl_model, train_loader, n_samples=n_samples, repeat=repeat)
    # Evaluate the CL model on the test set
    ssl_accs_test = ncc_evaluator.evaluate(ssl_model, test_loader, n_samples=n_samples, repeat=repeat)
    
    # Evaluate the NSCL (nscl_model) on the training set
    nscl_accs_train = ncc_evaluator.evaluate(nscl_model, train_loader, n_samples=n_samples, repeat=repeat)
    # Evaluate the NSCL model on the test set
    nscl_accs_test = ncc_evaluator.evaluate(nscl_model, test_loader, n_samples=n_samples, repeat=repeat)
    
    return ssl_accs_train, ssl_accs_test, nscl_accs_train, nscl_accs_test

def get_fewshot_loader(n_samples, dataset, labels,
                       classes=None,
                       batch_size=256,
                      ):
        """
        Extract n_samples per class from the training loader and return a DataLoader with only those samples.
        
        Args:
            n_samples (int): number of samples per class to extract.

        Returns:
            DataLoader: a new DataLoader with n_samples per class.
        """
        random.seed(123)

        class_to_indices = defaultdict(list)
        # Step 1: Collect all sample indices per class
        for idx in range(len(dataset)):
            label = labels[idx]

            class_to_indices[label].append(idx)

        # Step 2: Randomly sample n_samples from each class
        selected_indices = []
        for c in classes:
            indices = class_to_indices[c]
            if len(indices) < n_samples:
                raise ValueError(f"Not enough samples for class {c} (found {len(indices)}, needed {n_samples})")
            selected = random.sample(indices, n_samples)
            selected_indices.extend(selected)

        # Step 3: Create new dataloader from subset
        fewshot_subset = Subset(dataset, selected_indices)

        batch_size = min(len(selected_indices), batch_size)
        fewshot_loader = DataLoader(fewshot_subset, batch_size=batch_size,
                                    shuffle=True, drop_last=False)

        return fewshot_loader