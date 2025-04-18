import torch
from torch.utils.data import Dataset
from torchvision import  transforms

from typing import Union


class SimCLRDataset(Dataset):
    def __init__(self, dataset: Dataset,
                 train_transforms: transforms.Compose,
                 basic_transforms: transforms.Compose,
                 augment_both_views: bool = True,
                 dataset_name: str = 'imagenet'):
        
        self.dataset = dataset
        self.train_transforms = train_transforms
        self.basic_transforms = basic_transforms
        self.augment_both_views = augment_both_views
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        if 'cifar' in self.dataset_name:
            image, label = self.dataset[idx]
        elif 'imagenet' in self.dataset_name:
            image = self.dataset[idx]['image'].convert("RGB")
            label = self.dataset[idx]['label']

        view1 = self.train_transforms(image)
        if self.augment_both_views:
            view2 = self.train_transforms(image)
        else:
            view2 = self.basic_transforms(image)
            
        return view1, view2, label


