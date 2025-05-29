import numpy as np
import random
import wandb
import torch
from torch.utils.data import Subset, DataLoader
from torch.amp import autocast
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score

from tqdm import tqdm
from typing import List, Tuple, Optional
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
    - calculate accuracy rates
    - perform this for 'repeat' number of times
    """
    def __init__(self, train_loader: DataLoader,
                 train_labels: np.ndarray, 
                 output_classes:int =10 , 
                 device:str ='cuda',
                 labels=None):
        
        self.train_loader = train_loader
        self.train_labels = train_labels
        self.output_classes = output_classes
        self.device = device
        if labels is not None:
            self.label_map = self._label_map(labels)
        else:
            self.label_map = None

    def evaluate(self, model: torch.nn.Module, 
                 test_loader: torch.utils.data.DataLoader, 
                 n_samples:int = None,
                 repeat:int = 5,
                 embedding_layer:List[int] = [0,1]):
        """
        Args:
            N (int, required only for few-shot setting): Number of data points per class to calculate class centers.
            repeat (int, optional): Number of times to repeat the evaluation. Defaults to 5.
            embedding_layer (List[int], optional): List of embedding layers to use. Defaults to [-1].
    
        """
        model.eval()
        emb_dims = self._get_embedding_shapes(model, embedding_layer)
        num_embs = len(embedding_layer)
        
        # repeat the evaluation for 'repeat' number of times
        accs = []
        for r in range(repeat):
            print(f"Repeat {r+1}/{repeat} for NCCC evaluation")
            # calculate class centers
            means = self.fit(model, n_samples, num_embs, emb_dims, embedding_layer)

            # calculate NCCC score
            acc = self.calculate_nccc_score(num_embs, model, means, test_loader, embedding_layer)
            accs.append(acc)
            print("Accuracy: ", acc)

        # calculate average accuracy
        avg_accs = [sum([accs[i][j] for i in range(repeat)]) / repeat for j in range(num_embs)]
        return avg_accs
    

    def fit(self, model: torch.nn.Module,
            n_samples: Optional[int],
            num_embs: int,
            emb_dims: List[int],
            embedding_layer: List[int],):
        """
        fit the NCCC model
        - calculate class centers using N data points per class
        - store class centers
        """
        assert num_embs == len(embedding_layer)
        counts = [self.output_classes * [0] for _ in range(num_embs)] # tracks number of samples per class

        means = []
        for i in range(num_embs):
            means += [torch.zeros(self.output_classes, emb_dims[i]).to(self.device)]

        # if n_samples is not None, we need to sample n_samples per class
        if n_samples is not None:
            loader = self._get_fewshot_loader(n_samples)
        else:
            loader = self.train_loader

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Computing means for N samples = {n_samples}"):
                _, x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                if self.label_map is not None:
                    y = torch.tensor([self.label_map[i.item()] for i in y], device=self.device)

                # get the embeddings
                embeddings = self._extract_embeddings(model, x, embedding_layer)

                for i, emb in enumerate(embeddings):
                    for c in range(self.output_classes):
                        idxs = y == c
                        if len(idxs) == 0:
                            continue
                        h_c = emb[idxs]
                        means[i][c] += torch.sum(h_c, dim=0)
                        counts[i][c] += h_c.shape[0]

        # calculate the means
        for i in range(num_embs):
            for c in range(self.output_classes):
                means[i][c] /= max(counts[i][c], 1e-6)  # avoid division by zero

        return means
    
    @torch.no_grad()
    def calculate_nccc_score(self, num_embs, model, means, test_loader, embedding_layer):
        """
        calculate NCCC score
        """
        corrects = num_embs * [0.0]
        total = 0.0

        for batch in tqdm(test_loader, desc="Computing NCCC Score"):
            _, x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            if self.label_map is not None:
                y = torch.tensor([self.label_map[i.item()] for i in y], device=self.device)

            embeddings = self._extract_embeddings(model, x, embedding_layer)
            total += y.size(0)

            for i, emb in enumerate(embeddings):
                emb = emb.detach()
                # calculate the distance
                dist = torch.cdist(emb.unsqueeze(0), means[i].unsqueeze(0)).squeeze(0)
                preds = torch.argmin(dist, dim=1)
                corrects[i] += torch.sum(preds == y).item()

        dataset_size = len(test_loader.dataset)
        accs = [corrects[i] / dataset_size for i in range(num_embs)]

        return accs
    
    def _get_fewshot_loader(self, n_samples):
        """
        Extract n_samples per class from the training loader and return a DataLoader with only those samples.
        """
        dataset = self.train_loader.dataset

        class_to_indices = defaultdict(list)
        # Step 1: Collect all sample indices per class
        for idx in range(len(dataset)):
            label = self.train_labels[idx]
            class_to_indices[label].append(idx)

        # Step 2: Randomly sample n_samples from each class
        selected_indices = []
        for c in class_to_indices.keys():
            indices = class_to_indices[c]
            if len(indices) < n_samples:
                raise ValueError(f"Not enough samples for class {c} (found {len(indices)}, needed {n_samples})")
            selected = random.sample(indices, n_samples)
            selected_indices.extend(selected)

        # Step 3: Create new dataloader from subset
        fewshot_subset = Subset(dataset, selected_indices)
        batch_size = min(len(selected_indices), self.train_loader.batch_size)
        fewshot_loader = DataLoader(fewshot_subset, batch_size=batch_size,
                                    shuffle=True, drop_last=False,
                                    pin_memory=True, num_workers=32)
        return fewshot_loader
    
    def _label_map(self, labels):
        """
        map the labels to the output_classes
        """
        label_map = {}
        for i, label in enumerate(labels):
            label_map[label] = i

        return label_map
    
    def _extract_embeddings(self, model: torch.nn.Module, x: torch.Tensor,
                            embedding_layer: List[int]) -> List[torch.Tensor]:
        h, g_h = model(x)
        embeddings = [h.view(h.size(0), -1), g_h.view(g_h.size(0), -1)]
        return [embeddings[i] for i in embedding_layer]
    
    def _get_embedding_shapes(self, model: torch.nn.Module, embedding_layer: List[int]) -> List[int]:
        with torch.no_grad():
            _, x, _ = next(iter(self.train_loader))
            x = x.to(self.device)
            h, g_h = model(x[:1])
            embeddings = [h.view(1, -1), g_h.view(1, -1)]
            return [embeddings[i].size(1) for i in embedding_layer]


# ================= 3️⃣ Linear Probing ===================
class LinearProbeEval:
    def __init__(self, model, train_loader, 
                 output_classes=10, epochs=101, lr=3e-4, 
                 device='cuda', labels=None,
                 log_every=10,
                 log_to_wandb=False,
                 wandb_project=None,
                 wandb_name=None,
                 train_labels=None,
                 test_labels=None):
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
        self.train_labels = train_labels
        self.test_labels = test_labels

    def fit(self, loader, 
            linear_projs, optimizer, 
            embedding_layer=[0],
            test_loader=None,
            n_samples=None):
        
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
                embeddings = [h.detach(), g_h.detach()]

                loss = 0.0
                for i, j in enumerate(embedding_layer):
                    emb = embeddings[j].view(embeddings[j].shape[0], -1)
                    out = linear_projs[i](emb)
                    loss += self.loss_fn(out, y)

                loss.backward()
                optimizer.step()
                # print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
            
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
                 n_samples=None, repeat=1, 
                 embedding_layer=[0],
                 wandb_name=None):
        
        if wandb_name is not None:
            self.wandb_name = wandb_name
        
        results_train = []
        results_test = []

        for _ in range(repeat):
            # initialize linear probes and optimizer
            linear_probes, params = self._init_linear_probes(embedding_layer)
            print("Linear Probes initialized!")
            optimizer = torch.optim.Adam(params, lr=self.lr)
            
            if n_samples is not None:
                loader = self._get_fewshot_loader(n_samples)
                print("Few shot loader initialized!")
            else:
                loader = self.train_loader
                # repeat = 1 # enforce 1 in full-shot setting
            
            # fit on the current loader
            self.fit(loader, linear_probes, optimizer, embedding_layer, test_loader, n_samples)
            
            # evaluation loop
            tot_accs, _ = self._evaluate_accuracy(self.train_loader, linear_probes, embedding_layer)
            results_train.append(tot_accs)
            print(f"Train accuracy: {tot_accs}")
            if test_loader is not None:
                tot_accs_test, _ = self._evaluate_accuracy(test_loader, linear_probes, embedding_layer)
                results_test.append(tot_accs_test)
                print(f"Test accuracy: {tot_accs_test}")

        # average over repeats
        if repeat == 1:
            return results_train[0], results_test[0]
        else:
            avg_result_train = [sum(r[i] for r in results_train)/repeat for i in range(len(embedding_layer))]
            avg_result_test = [sum(r[i] for r in results_test)/repeat for i in range(len(embedding_layer))]

            return avg_result_train, avg_result_test

    def _init_linear_probes(self, embedding_layer):
        # Initialize a linear classifier for each embedding layer
        with torch.no_grad():
            x, _, _ = next(iter(self.train_loader))
            x = x.to(self.device)
            # just take a single sample
            x = x[0].unsqueeze(0)
            h, g_h = self.model(x)
            embeddings = [h, g_h]

        linear_probes = []
        params = []

        for i in embedding_layer:
            emb_dim = embeddings[i].view(embeddings[i].shape[0], -1).shape[1]
            probe = torch.nn.Linear(emb_dim, self.output_classes, bias=False).to(self.device)
            linear_probes.append(probe)
            params += list(probe.parameters())

        return linear_probes, params

    @torch.no_grad()
    def _evaluate_accuracy(self, loader, linear_projs, embedding_layer):
        self.model.eval()
        losses = [0 for _ in embedding_layer]
        corrects = [0 for _ in embedding_layer]
        total = 0

        with torch.no_grad():
            for batch in tqdm(loader):
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
        """
        random.seed(123)
        dataset = self.train_loader.dataset

        class_to_indices = defaultdict(list)
        # Step 1: Collect all sample indices per class
        for idx in range(len(dataset)):
            label = self.train_labels[idx]
            class_to_indices[label].append(idx)

        # Step 2: Randomly sample n_samples from each class
        selected_indices = []
        for c in class_to_indices.keys():
            indices = class_to_indices[c]
            if len(indices) < n_samples:
                raise ValueError(f"Not enough samples for class {c} (found {len(indices)}, needed {n_samples})")
            selected = random.sample(indices, n_samples)
            selected_indices.extend(selected)

        # Step 3: Create new dataloader from subset
        fewshot_subset = Subset(dataset, selected_indices)
        batch_size = min(len(selected_indices), self.train_loader.batch_size)
        fewshot_loader = DataLoader(fewshot_subset, batch_size=batch_size,
                                    shuffle=True, drop_last=False,
                                    pin_memory=True, num_workers=32)
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

class RepresentationEvaluator:
    def __init__(self, model, device='cuda', num_classes=10):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.model.eval()

    @torch.no_grad()
    def extract_features(self, loader, test=False):
        features_h = []
        features_gh = []
        labels = []

        for batch in tqdm(loader, desc="Extracting Features"):
            if test:
                x, y = batch
            else:
                _, x, y = batch # Assuming (index, data, label) format
            x = x.to(self.device)
            h, g_h = self.model(x)
            features_h.append(h.view(h.size(0), -1).cpu())
            features_gh.append(g_h.view(g_h.size(0), -1).cpu())
            labels.append(y.cpu())

        features_h = torch.cat(features_h, dim=0).numpy()
        features_gh = torch.cat(features_gh, dim=0).numpy()
        labels = torch.cat(labels, dim=0).numpy()

        return [features_h, features_gh], labels

    def compute_cdnv(self, features, labels):
        """
        Compute cdnv for each embedding separately (h and g_h)
        Inputs:
            - features: tensors of shape (num_samples, feature_dim)
            - labels: labels tensor of shape (num_samples,)
        Outputs:
            - cdnv: cdnv value for each embedding
        """

        device = self.device
        mean = [0.0] * self.num_classes
        mean_s = [0.0] * self.num_classes
       
        for c in range(self.num_classes):
            idxs = (labels==c).nonzero(as_tuple=True)[0]
            if len(idxs) == 0:
                continue

            feats_c = features[idxs]
            mean[c] = torch.sum(feats_c, dim=0) / len(idxs)
            mean_s[c] = torch.sum((feats_c**2)) / len(idxs)

        avg_cdnv = 0.0
        total_num_pairs = self.num_classes * (self.num_classes - 1) / 2

        for class1 in range(self.num_classes):
            for class2 in range(class1+1, self.num_classes):
                if mean[class1] is None or mean[class2] is None:
                    continue
                # variance = E[x^2] - (E[x])^2, computed as mean_s - mean**2
                variance1 = abs(mean_s[class1].item() - (mean[class1]**2).sum().item())
                variance2 = abs(mean_s[class2].item() - (mean[class2]**2).sum().item())
                variance_avg = (variance1 + variance2) / 2
                dist = torch.norm(mean[class1] - mean[class2])**2
                dist = dist.item()

                cdnv = variance_avg / dist
                avg_cdnv += cdnv / total_num_pairs

        return avg_cdnv
    
    def compute_directional_cdnv(self, features, labels, means=None):
        features = features.to(self.device)
        labels = labels.to(self.device)

        if means is None:
            means = self.compute_class_means(features, labels)
        
        avg_dir_cdnv = 0.0
        total_num_pairs = self.num_classes * (self.num_classes - 1)

        for class1 in range(self.num_classes):
            idxs = (labels == class1).nonzero(as_tuple=True)[0]
            if len(idxs) == 0:
                continue

            features1 = features[idxs]

            for class2 in range(self.num_classes):
                if class2 == class1:
                    continue

                v = means[class2] - means[class1]
                v_norm = v.norm()
                if v_norm == 0:
                    continue  # skip degenerate pair

                v_hat = v / v_norm
                projections = (features1 - means[class1]) @ v_hat
                dir_var = torch.mean(projections ** 2)
                dir_cdnv = dir_var / (v_norm ** 2)

                avg_dir_cdnv += dir_cdnv / total_num_pairs

        return avg_dir_cdnv.item()

    def compute_nccc(self, features, labels,
                     means=None):
        """
        Compute NCCC accuracy using precomputed features
        Inputs:
            - features: features tensor (either from h or g_h) of shape (num_samples, feature_dim)
            - labels: labels tensor of shape (num_samples,)
        Outputs:
            - accuracy: accuracy of nearest mean classifier
        """

        total_samples = labels.shape[0]

        # keep everything on the same device
        features = features.to(self.device)
        labels = labels.to(self.device)

        # Compute class means
        if means is None:
            # useful during training
            means = self.compute_class_means(features, labels)

        # Compute distances to class means
        dist = torch.cdist(features, means)
        preds = dist.argmin(dim=1)
        correct = preds.eq(labels).sum().item()
        accuracy = correct / total_samples
        return accuracy

    def compute_class_means(self, features, labels):
        """
        Computes class-wise means corresponding to the given embedding layer
        Inputs:
            - features: features tensor of shape (num_samples, feature_dim)
            - labels: labels tensor of shape (num_samples,)
        Outputs:
            - means: class means tensor of shape (num_classes, feature_dim)
        """

        means = torch.zeros(self.num_classes, features.size(1)).to(self.device)
        counts = torch.zeros(self.num_classes).to(self.device)

        for i in range(self.num_classes):
            idxs = (labels == i).nonzero(as_tuple=True)[0]
            if len(idxs) == 0:
                continue
            means[i] = features[idxs].mean(dim=0)

        return means

    def _test_nearest_mean(self, loader, means_h, means_gh):
        correct_h = 0
        correct_gh = 0
        total = 0

        for batch in loader:
            _, x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            h, g_h = self.model(x)

            h = h.view(h.size(0), -1)
            g_h = g_h.view(g_h.size(0), -1)

            dist_h = torch.cdist(h, means_h)
            preds_h = dist_h.argmin(dim=1)
            correct_h += preds_h.eq(y).sum().item()

            dist_gh = torch.cdist(g_h, means_gh)
            preds_gh = dist_gh.argmin(dim=1)
            correct_gh += preds_gh.eq(y).sum().item()

            total += y.size(0)

        acc_h = correct_h / total
        acc_gh = correct_gh / total
        return acc_h, acc_gh

# ================= 4️⃣ CDNV Evaluation ===================
@torch.no_grad()
def cal_cdnv(model, loader, settings):
    model.eval()

    initialized_results = False
    N = []
    mean = []
    mean_s = []
    cdnvs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="CDNV Eval Progress"):
            _, data, target = batch
            # if data.shape[0] != settings.batch_size:
            #     continue

            data, target = data.to(settings.device), target.to(settings.device)
            h, g_h = model(data)
            embeddings = [h, g_h]
            feature_dims = [h.shape[1], g_h.shape[1]]
            # embeddings = embeddings.unsqueeze(0)
            num_embs = len(embeddings)

            if initialized_results == False: # if we did not init means to zero
                N = [settings.num_output_classes*[0] for _ in range(num_embs)]
                mean = [settings.num_output_classes*[0] for _ in range(num_embs)]
                mean_s = [settings.num_output_classes*[0] for _ in range(num_embs)]
                initialized_results = True

            for i in range(num_embs): # which top layer to investigate
                z = embeddings[i]

                for c in range(settings.num_output_classes):
                    idxs = (target == c).nonzero(as_tuple=True)[0]
                    if len(idxs) == 0:  # If no class-c in this batch
                        continue

                    z_c = z[idxs, :]
                    mean[i][c] += torch.sum(z_c, dim=0)
                    N[i][c] += z_c.shape[0]
                    mean_s[i][c] += torch.sum(torch.square(z_c))
    
    for i in range(num_embs):
        for c in range(settings.num_output_classes):
            mean[i][c] /= N[i][c]
            mean_s[i][c] /= N[i][c]

        avg_cdnv = 0
        total_num_pairs = settings.num_output_classes * (settings.num_output_classes - 1) / 2
        for class1 in range(settings.num_output_classes):
            for class2 in range(class1 + 1, settings.num_output_classes):
                variance1 = abs(mean_s[i][class1].item() - torch.sum(torch.square(mean[i][class1])).item())
                variance2 = abs(mean_s[i][class2].item() - torch.sum(torch.square(mean[i][class2])).item())
                variance_avg = (variance1 + variance2) / 2
                dist = torch.norm((mean[i][class1]) - (mean[i][class2]))**2
                dist = dist.item()
                cdnv = variance_avg / dist
                avg_cdnv += cdnv / total_num_pairs

        cdnvs += [avg_cdnv]
    
    return cdnvs


@torch.no_grad()
def cal_directional_cdnv(model, loader, settings):
    model.eval()

    initialized_results = False
    N = []
    mean = []
    mean_s = []
    cdnvs = []
 
    with torch.no_grad():
        for batch in tqdm(loader, desc="CDNV Eval Progress"):
            _, data, target = batch
            # if data.shape[0] != settings.batch_size:
            #     continue

            data, target = data.to(settings.device), target.to(settings.device)
            h, g_h = model(data)
            embeddings = [h, g_h]
            # embeddings = embeddings.unsqueeze(0)
            num_embs = len(embeddings)

            if initialized_results == False: # if we did not init means to zero
                N = [settings.num_output_classes*[0] for _ in range(num_embs)]
                mean = [settings.num_output_classes*[0] for _ in range(num_embs)]
                mean_s = [settings.num_output_classes*[0] for _ in range(num_embs)]
                initialized_results = True

            for i in range(num_embs): # which top layer to investigate
                z = embeddings[i]

                for c in range(settings.num_output_classes):
                    idxs = (target == c).nonzero(as_tuple=True)[0]
                    if len(idxs) == 0:  # If no class-c in this batch
                        continue

                    z_c = z[idxs, :]
                    mean[i][c] += torch.sum(z_c, dim=0)
                    N[i][c] += z_c.shape[0]
                    mean_s[i][c] += torch.sum(torch.square(z_c))
    
    for i in range(num_embs):
        for c in range(settings.num_output_classes):
            mean[i][c] /= N[i][c]
            mean_s[i][c] /= N[i][c]

        avg_cdnv = 0
        total_num_pairs = settings.num_output_classes * (settings.num_output_classes - 1) / 2
        for class1 in range(settings.num_output_classes):
            for class2 in range(class1 + 1, settings.num_output_classes):
                variance1 = abs(mean_s[i][class1].item() - torch.sum(torch.square(mean[i][class1])).item())
                variance2 = abs(mean_s[i][class2].item() - torch.sum(torch.square(mean[i][class2])).item())
                variance_avg = (variance1 + variance2) / 2
                dist = torch.norm((mean[i][class1]) - (mean[i][class2]))**2
                dist = dist.item()
                cdnv = variance_avg / dist
                avg_cdnv += cdnv / total_num_pairs

        cdnvs += [avg_cdnv]
    
    return cdnvs


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