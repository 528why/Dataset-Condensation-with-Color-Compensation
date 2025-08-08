import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import kornia as K
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision

from scipy.ndimage.interpolation import rotate as scipyrotate
import math

from torch.utils.data import  Subset

from torch.optim.lr_scheduler import LambdaLR

from tqdm import tqdm
import torch.optim as optim
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, samples, transform=None, label_map=None):
        """
        samples: [(image_path, label), ...]
        transform: 可选的图像变换
        label_map: 可选的标签映射
        """
        self.samples = samples
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 获取样本路径和标签
        path, label = self.samples[idx]
        # 打开图像
        image = Image.open(path)
        # 应用变换
        if self.transform:
            image = self.transform(image)
        # 应用标签映射（如果存在）
        if self.label_map:
            label = self.label_map[label]
        return image, label


class SubsetWithSamples(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
    
    @property
    def samples(self):
        # 返回对应样本的图片路径和标签
        return [self.dataset.samples[i] for i in self.indices]
    
def get_dataset(args):
    print("args.dataset",args.dataset) 
    if args.dataset in ["imagenet",  "tinyimagenet", "cifar10", "cifar100", "syn_imagenette"]:
        train_dataset = datasets.ImageFolder(root=args.train_dir)
        idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
        if args.selection in ['DQ', 'Kmeans_SS','Kmeans_1','Kmeans_5','Kmeans_10','Kmeans_20','Random'] :
            indices = np.load(args.indices_path)
            train_dataset = SubsetWithSamples(train_dataset, indices)
        return train_dataset, idx_to_class, None
    elif args.dataset == 'imagenette' or args.dataset == "imagenet-woof":
        if args.dataset == 'imagenette':
            sub_imagenet_indices = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
        elif args.dataset == "imagenet-woof":
            sub_imagenet_indices = [193, 182, 258, 162, 155, 167, 159, 273, 207, 229]
        imagenet_dataset = datasets.ImageFolder(root=args.train_dir) 
        train_samples = [sample for sample in imagenet_dataset.samples if sample[1] in sub_imagenet_indices]
        # 构建 idx_to_class 映射
        idx_to_class = {i: imagenet_dataset.classes[sub_imagenet_indices[i]] for i in range(len(sub_imagenet_indices))}
        # 构建 label_map，用于将原始标签映射到子集标签
        label_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(sub_imagenet_indices)}
        train_dataset = CustomDataset(train_samples, transform=None, label_map=label_map)
        if args.selection in ['DQ', 'Kmeans_SS','Kmeans_1','Kmeans_5','Kmeans_10','Kmeans_20','Random'] :
            indices = np.load(args.indices_path)
            train_dataset = SubsetWithSamples(train_dataset, indices)
            print("get dataset:",len(train_dataset))
        print("label_map",label_map)
        return train_dataset, idx_to_class, label_map