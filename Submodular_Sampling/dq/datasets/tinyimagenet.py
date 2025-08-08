from torchvision import datasets, transforms
from torch import tensor, long
import os
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import Subset
import torch

class ImageFolderWithIndex(Dataset):
    def __init__(self, root, transform=None):
        # Load dataset using ImageFolder
        self.dataset = datasets.ImageFolder(root=root, transform=transform)
        self.samples = self.dataset.samples
        self.targets = torch.tensor(self.dataset.targets)
        # Add class names attribute
        self.classes = self.dataset.classes

    def __getitem__(self, idx):
        # Get sample and label
        sample, target = self.dataset[idx]
        # Return sample, label and index
        return sample, target, idx

    def __len__(self):
        return len(self.dataset)


def tinyimagenet(data_path):
    channel = 3
    im_size = (224, 224)
    num_classes = 200
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)

    # Load test dataset
    dst_test = datasets.ImageFolder(root=os.path.join(data_path, 'validation_by_class'), transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))


    dst_train = ImageFolderWithIndex(
    root=os.path.join(data_path, 'train_by_class'),
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
)

    class_names = dst_train.classes
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test