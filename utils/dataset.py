import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SimCLRDataset(Dataset):
    """
    Dataset wrapper for SimCLR-style contrastive learning.

    Each sample yields two augmented views of the same image along with its label.
    
    Attributes:
        dataset (Dataset): Base dataset (e.g., CIFAR, ImageNet).
        train_transforms (transforms.Compose): Data augmentation pipeline for contrastive learning.
        basic_transforms (transforms.Compose): Standard preprocessing pipeline for view2 if `augment_both_views` is False.
        augment_both_views (bool): If True, applies `train_transforms` to both views. If False, applies `basic_transforms` to the second view.
        dataset_name (str): Used for dataset-specific unpacking logic (e.g., dict vs tuple).
    """
    
    def __init__(
        self,
        dataset: Dataset,
        train_transforms: transforms.Compose,
        basic_transforms: transforms.Compose,
        augment_both_views: bool = True,
        dataset_name: str = 'imagenet'
    ):
        self.dataset = dataset
        self.train_transforms = train_transforms
        self.basic_transforms = basic_transforms
        self.augment_both_views = augment_both_views
        self.dataset_name = dataset_name.lower()

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int):
        """
        Returns two augmented views of the image and its label.

        Args:
            idx (int): Index of the sample.

        Returns:
            view1 (Tensor): First augmented view of the image.
            view2 (Tensor): Second view (augmented or basic) of the image.
            label (int): Label of the image.
        """
        if 'imagenet' in self.dataset_name:
            sample = self.dataset[idx]
            image = sample['image'].convert("RGB")
            label = sample['label']
        else:
            image, label = self.dataset[idx]

        view1 = self.train_transforms(image)
        view2 = self.train_transforms(image) if self.augment_both_views else self.basic_transforms(image)
        
        return view1, view2, label
