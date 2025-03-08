import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import datasets, transforms

from utils.augmentations import get_transforms
from utils.dataset import SimCLRDataset


def get_dataset(dataset_name, dataset_path,
                augment_both_views=True,
                batch_size=64, num_workers=4, 
                shuffle=True, **kwargs):
    
    multi_gpu = kwargs.pop('multi_gpu', False)
    world_size = kwargs.pop('world_size', 1)
    test = kwargs.pop('test', None)

    if dataset_name is None:
        # default to cifar10
        dataset_name = 'cifar10'

    if dataset_name == 'imagenet':
        dataset = datasets.ImageFolder(root=dataset_path)
        
    elif dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root=dataset_path, train=True, 
                                          download=True, transform=None)
        test_dataset = datasets.CIFAR10(root=dataset_path, train=False,
                                       download=True, transform=None)
    elif dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(root=dataset_path, train=True, 
                                         download=True, transform=None)
    else:
        raise NotImplementedError(f'no known dataset named {dataset_name}')
    
    train_transforms, basic_transforms = get_transforms('cifar')
    train_dataset = SimCLRDataset(train_dataset, 
                                train_transforms, basic_transforms,
                                augment_both_views=augment_both_views)
    
    # Adjust for multi-GPU
    shuffle = not multi_gpu  # Ensures DistributedSampler handles shuffling
    effective_batch_size = batch_size // world_size if multi_gpu else batch_size
    drop_last = multi_gpu  # Avoids uneven batches in DDP

    sampler = DistributedSampler(train_dataset, num_replicas=world_size) if multi_gpu else None

    train_dataloader = DataLoader(train_dataset, batch_size=effective_batch_size,
                                  shuffle=shuffle, num_workers=num_workers,
                                  pin_memory=True, drop_last=drop_last, 
                                  sampler=sampler)
    if test is not None:
        test_dataset = SimCLRDataset(test_dataset,
                                     train_transforms, basic_transforms,
                                     augment_both_views=False
                                     )
        test_dataloader = DataLoader(test_dataset, batch_size=effective_batch_size,
                                     shuffle=False, num_workers=num_workers,
                                     pin_memory=True)
        return train_dataset, train_dataloader, test_dataset, test_dataloader
    
    return train_dataset, train_dataloader

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

