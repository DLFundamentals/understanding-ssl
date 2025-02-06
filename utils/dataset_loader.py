import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from utils.augmentations import get_transforms


def get_dataset(dataset_name, dataset_path,
                       batch_size=64, num_workers=4, shuffle=True):

    if dataset_name is None:
        # default to cifar
        dataset_name = 'cifar'

    if dataset_name == 'imagenet':
        train_dataset = datasets.ImageFolder(root=dataset_path)
    elif dataset_name == 'cifar':
        train_dataset = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=transforms.ToTensor())
    else:
        raise NotImplementedError(f'no known dataset named {dataset_name}')
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers,
                                   pin_memory=True, drop_last=True)
    
    return train_dataset,  train_dataloader

def simclr_collate_fn(batch):
    # batch is a list of samples (each sample might be a PIL image)
    view1 = []
    view2 = []
    simclr_transform = get_transforms('simclr')
    for sample in batch:
        # Apply the transform twice to the same sample
        view1.append(simclr_transform(sample))
        view2.append(simclr_transform(sample))
    # Stack them into tensors (assumes each transform returns a tensor)
    return torch.stack(view1), torch.stack(view2)

