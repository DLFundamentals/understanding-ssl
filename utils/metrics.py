import numpy as np
import torch
from torch.amp import autocast
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score

from tqdm import tqdm
from typing import List, Tuple
from collections import defaultdict

# =================1️⃣ KNN Evaluation =================
class KNN:
    def __init__(self, model, k, device = 'cuda'):
        self.model = model
        self.k = k
        self.device = device

        # set model to eval
        self.model.eval()

    def extract_features(self, loader):
        "Extract features from a trained model"
        x_lst, features, label_lst = [], [], []

        with torch.no_grad():
            for batch in tqdm(loader):
                _, x, label = batch
                x = x.to(self.device)
                with autocast(device_type='cuda'):
                    # forward pass
                    _, z = self.model(x)

                # store features to cpu
                features.append(z.cpu())
                label_lst.append(label.cpu())

        features = torch.cat(features, dim = 0)
        label_lst = torch.cat(label_lst, dim = 0)

        return features, label_lst
    
    def knn_eval(self, train_loader, test_loader=None):
        "Evaluates KNN accuracy in feature space"
        z_train, y_train = self.extract_features(train_loader)
        features_np = z_train.numpy()
        labels_np = y_train.numpy()

        # look for NAN values
        if np.isnan(features_np).any():
            print("NaN values found in features. Replacing with 0")
            features_np = np.nan_to_num(features_np)
            
        if isinstance(self.k, int):
            knn = KNeighborsClassifier(n_neighbors = self.k, metric="cosine").fit(features_np, labels_np)
            train_acc = 100 * np.mean(cross_val_score(knn, features_np, labels_np, cv=5))
            print(f"KNN Evaluation: Train Acc: {train_acc:.2f}%")

            if test_loader:
                z_test, y_test = self.extract_features(test_loader)
                features_test_np = z_test.numpy()
                labels_test_np = y_test.numpy()

                test_acc = 100 * knn.score(features_test_np, labels_test_np)
                print(f"KNN Evaluation: Test Acc: {test_acc:.2f}%")
                return train_acc, test_acc
            
            return train_acc, None

        elif isinstance(self.k, list):
            train_acc = []
            test_acc = []
            for k in self.k:
                knn = KNeighborsClassifier(n_neighbors = k, metric="cosine").fit(features_np, labels_np)
                train_acc_k = 100 * np.mean(cross_val_score(knn, features_np, labels_np, cv=5))
                print(f"Train Accuracy for k={k}: {train_acc_k:.2f}")
                train_acc.append(train_acc_k)

                if test_loader:
                    z_test, y_test = self.extract_features(test_loader)
                    features_test_np = z_test.numpy()
                    labels_test_np = y_test.numpy()

                    test_acc_k = 100 * knn.score(features_test_np, labels_test_np)
                    print(f"Test Accuracy for k={k}: {test_acc_k:.2f}")
                    test_acc.append(test_acc_k)

            return train_acc, test_acc


# =================2️⃣ NCCC Evaluation =================
class NCCCEval:
    """
    perform NCCC evaluation in a normal or few-shot setting
    - calculate class-center using N data points per class
    - calculate NCCC score for each class
    - calucate accuracy rates
    - perform this for 'repeat' number of times
    """
    def __init__(self, train_loader, output_classes=10 , device='cuda'):
        self.train_loader = train_loader
        self.output_classes = output_classes
        self.device = device
    

    def evaluate(self, model: torch.nn.Module, 
                 test_loader: torch.utils.data.DataLoader, 
                 n_samples:int = None, repeat:int =5,
                 embedding_layer:List[int] = [0,1]):
        """
        Args:
            N (int, required only for few-shot setting): Number of data points per class to calculate class centers.
            repeat (int, optional): Number of times to repeat the evaluation. Defaults to 5.
            embedding_layer (List[int], optional): List of embedding layers to use. Defaults to [-1].
    
        """
        model.eval()

        batch = next(iter(self.train_loader))
        _, x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        # get the embedding layer
        h, g_h = model(x)
        embeddings = [h, g_h]

        # select the embedding layer
        embeddings = [embeddings[i] for i in embedding_layer]
        num_embs = len(embeddings)
        emb_dims = []
        for emb in embeddings:
            emb = emb.view(emb.shape[0], -1)
            emb_dims.append(emb.shape[1])
        

        # repeat the evaluation for 'repeat' number of times
        accs = []
        for _ in range(repeat):
            # calculate class centers
            means = self.fit(model, n_samples, num_embs, emb_dims, embedding_layer)

            # calculate NCCC score
            acc = self.calculate_nccc_score(num_embs, model, means, test_loader, embedding_layer)
            accs.append(acc)

        # calculate average accuracy
        avg_accs = [sum([accs[i][j] for i in range(repeat)]) / repeat for j in range(num_embs)]

        return avg_accs
    

    def fit(self, model, n_samples,
            num_embs, emb_dims, embedding_layer,):
        """
        fit the NCCC model
        - calculate class centers using N data points per class
        - store class centers
        """
        assert num_embs == len(embedding_layer)
        N = [self.output_classes * [0] for _ in range(num_embs)] # tracks number of samples per class

        means = []
        for i in range(num_embs):
            means += [torch.zeros(self.output_classes, emb_dims[i]).to(self.device)]

        with torch.no_grad():
            for batch in self.train_loader:
                _, x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                if n_samples is not None:
                    # select indxs for inputs based on N and output_classes
                    final_mask = self.get_batch_idx_mask(y, n_samples, N)
                    x = x[final_mask]
                    y = y[final_mask]
                
                # non-empty x
                if x.shape[0] == 0:
                    continue

                # get the embeddings
                h, g_h = model(x)
                embeddings = [h, g_h]

                for i in embedding_layer:
                    emb = embeddings[i].view(embeddings[i].shape[0], -1)
                    
                    for c in range(self.output_classes):
                        idxs = y == c
                        if len(idxs) == 0:
                            continue

                        h_c = emb[idxs]
                        means[i][c] += torch.sum(h_c, dim=0)
                        N[i][c] += h_c.shape[0]

        # calculate the means
        for i in range(num_embs):
            for c in range(self.output_classes):
                means[i][c] /= N[i][c]

        return means
         
    def get_batch_idx_mask(self, y, n_samples, N):
        final_mask = torch.zeros_like(y, dtype=torch.bool)  # Use boolean mask for indexing
        for c in range(self.output_classes):
            mask = (y == c).nonzero(as_tuple=True)[0]  # Indices where class == c

            if N[0][c] >= n_samples:  # If we already have enough, skip
                continue

            n_remaining = n_samples - N[0][c]
            available_samples = mask.shape[0]

            if available_samples > 0:
                # If we have more than needed, randomly select `n_remaining ± small variation`
                if available_samples >= n_remaining:
                    random_offset = torch.randint(-2, 3, (1,)).item()  # Small variation of ±2
                    num_to_take = max(1, min(n_remaining + random_offset, available_samples))
                else:
                    num_to_take = available_samples  # Take whatever is available

                selected_indices = mask[torch.randperm(available_samples)[:num_to_take]]
                final_mask[selected_indices] = True
                N[0][c] += num_to_take  # Update count

        return final_mask

    
    @torch.no_grad()
    def calculate_nccc_score(self, num_embs, model, means, test_loader, embedding_layer):
        """
        calculate NCCC score
        """
        corrects = num_embs * [0.0]

        for batch in tqdm(test_loader):
            _, x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            h, g_h = model(x)
            embeddings = [h, g_h]

            for i in embedding_layer:
                emb = embeddings[i].view(embeddings[i].shape[0], -1)
                emb = emb.detach()

                # calculate the distance
                dist = torch.cdist(emb.unsqueeze(0), means[i].unsqueeze(0)).squeeze(0)
                preds = torch.argmin(dist, dim=1)
                corrects[i] += torch.sum(preds == y).item()

        dataset_size = len(test_loader.dataset)
        accs = [corrects[i] / dataset_size for i in range(num_embs)]

        return accs
                
        

# ================= 3️⃣ CDNV Evaluation =================


# ================= 4️⃣ Anisotropy Evaluation =================
@torch.no_grad()
def anisotropy(model, loader, 
               output_classes=10, embedding_layer=1,
               device='cuda'):
    """
    Calculate the anisotropy of the data:

                    anisotropy = λ_max / max(λ_min, ε)

    where λmax, λmin are the max/min eigenvalues of the covariance matrix of the data
    """
    model.eval()
    inputs = defaultdict(list)
    for batch in loader:
        _, x, y = batch
        h, g_h = model(x.to(device))
        embeddings = [h, g_h]

        # store the inputs class-wise
        for i in range(output_classes):
            idxs = y == i
            if torch.sum(idxs) == 0:
                continue
            inputs[i].append(embeddings[embedding_layer][idxs])

    anisotropies = []

    for i in range(output_classes):
        if len(inputs[i]) == 0:
            anisotropies.append(float('nan'))  # Handle missing classes properly
            continue

        # Concatenate embeddings for class i
        class_embeddings = torch.cat(inputs[i], dim=0)  # Shape: (N, D)

        # Compute covariance matrix (D, D)
        cov_matrix = torch.cov(class_embeddings.T)

        # Compute eigenvalues
        eigvals = torch.linalg.eigvalsh(cov_matrix)

        # Compute anisotropy ratio with numerical stability
        anisotropy_value = eigvals[-1] / max(eigvals[0], 1e-6)
        anisotropies.append(anisotropy_value)

    return anisotropies