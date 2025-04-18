import numpy as np
import random
import wandb
import torch
from torch.utils.data import Subset, DataLoader
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
    def __init__(self, train_loader, output_classes=10 , device='cuda',
                 labels=None):
        self.train_loader = train_loader
        self.output_classes = output_classes
        self.device = device
        if labels is not None:
            self.label_map = self._label_map(labels)
        else:
            self.label_map = None
    

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
        if self.label_map is not None:
            y = torch.tensor([self.label_map[i.item()] for i in y], device=self.device)

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
                if self.label_map is not None:
                    y = torch.tensor([self.label_map[i.item()] for i in y], device=self.device)

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
            if self.label_map is not None:
                y = torch.tensor([self.label_map[i.item()] for i in y], device=self.device)

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
    
    def _label_map(self, labels):
        """
        map the labels to the output_classes
        """
        label_map = {}
        for i, label in enumerate(labels):
            label_map[label] = i

        return label_map
                


# ================= 3️⃣ Linear Probing ===================
class LinearProbeEval:
    def __init__(self, model, train_loader, 
                 output_classes=10, epochs=101, lr=3e-4, 
                 device='cuda', labels=None,
                 log_every=10,
                 log_to_wandb=False,
                 wandb_project=None,
                 wandb_name=None):
        self.model = model
        self.model.eval()

        self.train_loader = train_loader
        self.output_classes = output_classes
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.label_map = self._label_map(labels) if labels is not None else None
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.log_every = log_every
        self.log_to_wandb = log_to_wandb
        self.wandb_initialized = False
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
        self.wandb_defined = False

    def fit(self, loader, 
            linear_projs, optimizer, 
            embedding_layer=[0],
            test_loader=None,
            n_samples=1000):
        
        if n_samples==1000:
            print("")
        # training loop
        for epoch in tqdm(range(self.epochs), desc=f'N samples = {n_samples}'):
            
            for proj in linear_projs:
                proj.train()

            for batch in loader:
                _, x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                if self.label_map:
                    y = torch.tensor([self.label_map[i.item()] for i in y], device=self.device)

                optimizer.zero_grad()
                with torch.no_grad():
                    h, g_h = self.model(x)
                embeddings = [h, g_h]

                loss = 0.0
                for i, j in enumerate(embedding_layer):
                    emb = embeddings[j].view(embeddings[j].shape[0], -1)
                    out = linear_projs[i](emb)
                    loss += self.loss_fn(out, y)

                loss.backward()
                optimizer.step()
            
            # 🔁 Log to wandb
            if self.log_to_wandb and (epoch%self.log_every==0):
                if not self.wandb_initialized:
                    wandb.init(project=self.wandb_project or "linear-probe-eval",
                            name=self.wandb_name, reinit=True)
                    self.wandb_initialized = True

                tot_accs, tot_losses = self._evaluate_accuracy(self.train_loader, linear_projs, embedding_layer)
                print(f"Train accuracy: {tot_accs}")
                self.log_metrics(tot_accs, tot_losses, epoch, self.wandb_defined)
                if test_loader is not None:
                    tot_accs, tot_losses = self._evaluate_accuracy(test_loader, linear_projs, embedding_layer)
                    print(f"Test accuracy: {tot_accs}")
                    self.log_metrics_test(tot_accs, tot_losses, epoch, self.wandb_defined)
                
                self.wandb_defined = True


    def evaluate(self, test_loader=None, 
                 n_samples=None, repeat=1, embedding_layer=[0],
                 wandb_name=None):
        
        if wandb_name is not None:
            self.wandb_name = wandb_name
        
        results = []
        results_test = []

        for _ in range(repeat):
            # initialize linear probes and optimizer
            linear_projs, params = self._init_linear_projs(embedding_layer)
            optimizer = torch.optim.Adam(params, lr=self.lr)
            

            if n_samples is not None:
                loader = self._get_fewshot_loader(n_samples)
            else:
                loader = self.train_loader
                # repeat = 1 # enforce 1 in full-shot setting
            
            # fit on the current loader
            self.fit(loader, linear_projs, optimizer, embedding_layer, test_loader, n_samples)
            
            # evaluation loop
            tot_accs, _ = self._evaluate_accuracy(self.train_loader, linear_projs, embedding_layer)
            results.append(tot_accs)
            if test_loader is not None:
                tot_accs_test, _ = self._evaluate_accuracy(test_loader, linear_projs, embedding_layer)
                results_test.append(tot_accs_test)

        # average over repeats
        if repeat == 1:
            return results[0], results_test[0]
        else:
            avg_result = [sum(r[i] for r in results)/repeat for i in range(len(embedding_layer))]
            avg_result_test = [sum(r[i] for r in results_test)/repeat for i in range(len(embedding_layer))]
            return avg_result, avg_result_test

    def _init_linear_projs(self, embedding_layer):
        # Initialize a linear classifier for each embedding layer
        with torch.no_grad():
            x, _, _ = next(iter(self.train_loader))
            x = x.to(self.device)
            h, g_h = self.model(x)
            embeddings = [h, g_h]

        linear_projs = []
        params = []

        for i in embedding_layer:
            emb_dim = embeddings[i].view(embeddings[i].shape[0], -1).shape[1]
            proj = torch.nn.Linear(emb_dim, self.output_classes, bias=False).to(self.device)
            linear_projs.append(proj)
            params += list(proj.parameters())

        return linear_projs, params

    @torch.no_grad()
    def _evaluate_accuracy(self, loader, linear_projs, embedding_layer):
        self.model.eval()
        losses = [0 for _ in embedding_layer]
        corrects = [0 for _ in embedding_layer]
        total = 0

        with torch.no_grad():
            for batch in loader:
                _, x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                if self.label_map:
                    y = torch.tensor([self.label_map[i.item()] for i in y], device=self.device)

                h, g_h = self.model(x)
                embeddings = [h, g_h]
                total += y.size(0)

                for i, j in enumerate(embedding_layer):
                    emb = embeddings[j].view(embeddings[j].shape[0], -1)
                    out = linear_projs[i](emb)
                    losses[i] += self.loss_fn(out, y).item()
                    preds = torch.argmax(out, dim=1)
                    corrects[i] += (preds == y).sum().item()

        tot_accs = [c / total for c in corrects]
        tot_losses = [l/total for l in losses]

        return tot_accs, tot_losses

    def _label_map(self, labels):
        return {label: idx for idx, label in enumerate(labels)}

    def _get_fewshot_loader(self, n_samples):
        """
        Extract n_samples per class from the training loader and return a DataLoader with only those samples.
        
        Args:
            n_samples (int): number of samples per class to extract.

        Returns:
            DataLoader: a new DataLoader with n_samples per class.
        """
        random.seed(123)
        dataset = self.train_loader.dataset
        class_to_indices = defaultdict(list)
        # Step 1: Collect all sample indices per class
        for idx in range(len(dataset)):
            sample = dataset[idx]
            _, _, label = sample

            class_to_indices[label].append(idx)

        # Step 2: Randomly sample n_samples from each class
        selected_indices = []
        for c in range(self.output_classes):
            indices = class_to_indices[c]
            if len(indices) < n_samples:
                raise ValueError(f"Not enough samples for class {c} (found {len(indices)}, needed {n_samples})")
            selected = random.sample(indices, n_samples)
            selected_indices.extend(selected)

        # Step 3: Create new dataloader from subset
        fewshot_subset = Subset(dataset, selected_indices)

        batch_size = min(len(selected_indices), self.train_loader.batch_size)
        fewshot_loader = DataLoader(fewshot_subset, batch_size=batch_size,
                                    shuffle=True, drop_last=False)

        return fewshot_loader


    def log_metrics(self, acc_rates, losses, epoch, wandb_defined=False):
        if not wandb_defined:
            wandb.define_metric("epoch")
            num_embs = len(acc_rates)
            for i in range(num_embs):
                wandb.define_metric(f"train_accuracy_{i}", step_metric="epoch")
                wandb.define_metric(f"lin_prob_loss_{i}", step_metric="epoch")

        log_data = defaultdict()

        log_data["epoch"] = epoch
        num_embs = len(acc_rates)
        for i in range(num_embs):
            log_data[f"train_accuracy_{i}"] = acc_rates[i]
            log_data[f"lin_prob_loss_{i}"] = losses[i]

        wandb.log(log_data)
        
    def log_metrics_test(self, acc_rates, losses, epoch, wandb_defined=False):
        if not wandb_defined:
            wandb.define_metric("epoch")
            num_embs = len(acc_rates)
            for i in range(num_embs):
                wandb.define_metric(f"test_accuracy_{i}", step_metric="epoch")
                wandb.define_metric(f"test_lin_prob_loss_{i}", step_metric="epoch")

        log_data = defaultdict()

        log_data["epoch"] = epoch
        num_embs = len(acc_rates)
        for i in range(num_embs):
            log_data[f"test_accuracy_{i}"] = acc_rates[i]
            log_data[f"test_lin_prob_loss_{i}"] = losses[i]

        wandb.log(log_data)

# ================= 4️⃣ CDNV Evaluation ===================






# ================= 5️⃣ Anisotropy Evaluation =================
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