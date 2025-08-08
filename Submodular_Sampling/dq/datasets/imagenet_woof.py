from torchvision import datasets, transforms
from torch import tensor, long
import os
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import Subset
import torch

class CustomSubsetDataset(Dataset):
    def __init__(self, dataset, indices, class_indices):
        self.dataset = dataset
        self.indices = indices
        self.class_indices = class_indices
        # Create label mapping
        self.class_map = {old: new for new, old in enumerate(class_indices)}
        self.classes = [dataset.classes[i] for i in class_indices]
        self.targets = torch.tensor([self.remap_labels(dataset.targets[i]) for i in indices])
    def remap_labels(self, original_target):
    # If the original label is an integer type, use it directly; if it's a tensor, call the item() method
        original_target = original_target.item() if isinstance(original_target, torch.Tensor) else original_target
        return self.class_map[original_target]

    def __getitem__(self, idx):
        # Get original sample and label
        sample, original_target = self.dataset[self.indices[idx]]
        # Return sample and mapped label
        mapped_target = self.remap_labels(original_target)
        return sample, mapped_target, idx

    def __len__(self):
        return len(self.indices)


def imagenet_woof(data_path):
    channel = 3
    im_size = (224, 224)
    num_classes = 10
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)
    sub_imagenet_indices = [193, 182, 258, 162, 155, 167, 159, 273, 207, 229]

    # Load test dataset
    imagenet_dataset_test = datasets.ImageFolder(root=os.path.join(data_path, 'val'), transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    test_targets = torch.tensor(imagenet_dataset_test.targets)
    test_samples = [i for i, target in enumerate(test_targets) if target in sub_imagenet_indices]
    dst_test = CustomSubsetDataset(imagenet_dataset_test, test_samples, sub_imagenet_indices)
    

    # Load training dataset
    imagenet_dataset_train = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    train_targets = torch.tensor(imagenet_dataset_train.targets)
    train_samples = [i for i, target in enumerate(train_targets) if target in sub_imagenet_indices]
    dst_train = CustomSubsetDataset(imagenet_dataset_train, train_samples, sub_imagenet_indices)

    class_names = dst_train.classes
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test