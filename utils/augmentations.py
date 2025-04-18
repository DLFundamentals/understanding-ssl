import torch
import torchvision.transforms as transforms

def get_transforms(dataset = 'imagenet'):

    if dataset == 'imagenet':
        s = 1.0
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur((3, 3), (1.5, 1.5))], p=0.1),
            transforms.ToTensor(),
            RepeatChannelsIfNeeded(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        basic_transform = transforms.Compose([
            transforms.Resize(256),                # resize short side to 256
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            RepeatChannelsIfNeeded(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    elif dataset == 'cifar' or 'cifar' in dataset:
        s = 0.5
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])
        basic_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])

    else:
        raise NotImplementedError(f'no known dataset named {dataset}')
    
    return train_transform, basic_transform

class RepeatChannelsIfNeeded:
    def __call__(self, img_tensor):
        if img_tensor.shape[0] == 1:
            print(f"Tensor shape before normalize: {img_tensor.shape}")
            return img_tensor.repeat(3, 1, 1)
        return img_tensor