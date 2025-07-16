import torch
from torchvision import transforms

class RepeatChannelsIfNeeded:
    """
    Transform that repeats a single-channel tensor to three channels.
    Used to handle grayscale images in datasets expecting RGB input.
    """
    def __call__(self, img_tensor):
        if img_tensor.shape[0] == 1:
            print(f"Repeating grayscale tensor with shape {img_tensor.shape}")
            return img_tensor.repeat(3, 1, 1)
        return img_tensor


def get_transforms(dataset: str = 'imagenet'):
    """
    Returns data augmentation (train) and basic evaluation transforms for a given dataset.

    Args:
        dataset (str): Name of the dataset. Options: 'imagenet', 'cifar', 'svhn'.

    Returns:
        train_transform (torchvision.transforms.Compose): Data augmentation pipeline.
        basic_transform (torchvision.transforms.Compose): Standard test/eval pipeline.
    """
    dataset = dataset.lower()
    
    if dataset == 'imagenet':
        s = 1.0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(1.5, 1.5))], p=0.1),
            transforms.ToTensor(),
            RepeatChannelsIfNeeded(),
            transforms.Normalize(mean=mean, std=std),
        ])
        basic_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            RepeatChannelsIfNeeded(),
            transforms.Normalize(mean=mean, std=std),
        ])

    elif 'cifar' in dataset or dataset == 'svhn':
        s = 0.5
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        basic_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset}")

    return train_transform, basic_transform
