import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch import tensor, long

class ImageFolderWithIndex(Dataset):
    def __init__(self, root, transform=None):
        # Load dataset using ImageFolder
        self.dataset = datasets.ImageFolder(root=root, transform=transform)
        self.samples = self.dataset.samples
        self.targets = torch.tensor(self.dataset.targets)
        # Add class names attribute
        self.classes = self.dataset.classes
        # Keep transform in dataset
        self.transform = transform

    def __getitem__(self, idx):
        # Get sample and label
        sample, target = self.dataset[idx]
        
        # If sample is already a Tensor, skip transform
        if isinstance(sample, torch.Tensor):
            return sample, target, idx
        
        # If transform exists, apply it
        if self.transform:
            sample = self.transform(sample)
        
        # Return sample, label and index
        return sample, target, idx

    def __len__(self):
        return len(self.dataset)


def CIFAR10(data_path):
    channel = 3
    im_size = (32, 32)
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = ImageFolderWithIndex(root='/data/cifar10/train_by_class', transform=transform)
    dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
