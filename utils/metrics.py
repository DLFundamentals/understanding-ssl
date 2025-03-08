import numpy as np
import torch
from torch.amp import autocast
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score

from tqdm import tqdm

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
