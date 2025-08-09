import os
import shutil
import argparse
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
# Set up argument parser
parser = argparse.ArgumentParser(description='Process ImageNet-subset dataset.')
parser.add_argument('--dataset', choices=['train', 'validation'], required=True, help='Specify whether to process train or validation dataset')
parser.add_argument('--ipc', type=int, required=True, help='ipc')
parser.add_argument('--subset', type=str, required=True, help='datset name')
parser.add_argument('--combine_mode', type=str, required=True, choices=['gradient', 'random', 'grid','fourfold_view'],
                        help='image combine_mode, choose either "gradient" or "random"')
args = parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self, samples, transform=None, label_map=None):
        self.samples = samples
        self.transform = transform
        self.label_map = label_map
        # If label_map exists, use the mapping, otherwise use original labels
        if label_map:
            self.classes = [str(label_map[key]) for key in sorted(label_map.keys())]
        else:
            original_labels = set([sample[1] for sample in samples])
            self.classes = [str(label) for label in sorted(original_labels)]
            
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.transform(Image.open(path).convert('RGB')) if self.transform else Image.open(path).convert('RGB')
        target = int(target)
        target = self.label_map.get(target, target)  # If mapping exists, replace with new label
        return sample, target

    def __len__(self):
        return len(self.samples)


if args.dataset == 'train':
    original_train_dir = f'../../DC3_ColorCompensation/compensated_{args.combine_mode}_{10}_{args.subset}_{args.ipc}'
    new_train_dir = f'../../DC3_ColorCompensation/{args.subset}_{args.combine_mode}_{10}/ipc{args.ipc}_train_by_class'

# Ensure new folder exists, create if it doesn't exist
if not os.path.exists(new_train_dir):
    os.makedirs(new_train_dir)
if args.dataset == "train":
    # Load dataset using ImageFolder
    dataset = datasets.ImageFolder(root=original_train_dir if args.dataset == 'train' else original_train_dir, transform=None)

    # Get label to class mapping (ImageFolder automatically handles class indices)
    class_to_idx = dataset.class_to_idx

    # Iterate through each class, copy images to new class subfolders
    for class_name, class_idx in class_to_idx.items():
        # Create new directory path
        new_class_folder_path = os.path.join(new_train_dir, f"{class_idx:05d}")
        if not os.path.exists(new_class_folder_path):
            os.makedirs(new_class_folder_path)
        
        # Iterate through all samples in the dataset, copy images belonging to current class
        for img_path, label in dataset.samples:
            if label == class_idx:  # If label matches
                shutil.copy(img_path, new_class_folder_path)

if args.dataset == 'validation':
    # Load dataset using ImageFolder
    dataset = datasets.ImageFolder(root=original_train_dir, transform=None)
    class_to_idx = dataset.class_to_idx
    if args.subset == "imagenette":
    # Only select the specified 10 classes
        sub_imagenet_indices = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
    elif args.subset == "imagenet-woof":
        sub_imagenet_indices = [193, 182, 258, 162, 155, 167, 159, 273, 207, 229]
    # Filter out samples belonging to these 10 classes
    imagenet_filtered = [sample for sample in dataset.samples if sample[1] in sub_imagenet_indices]
    # Update label mapping
    idx_to_class = {i: dataset.classes[sub_imagenet_indices[i]] for i in range(len(sub_imagenet_indices))}
    label_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(sub_imagenet_indices)}
    # Create validation set using custom label mapping
    dataset = CustomDataset(imagenet_filtered, transform=None, label_map=label_map)
    
    # Report newly created folders
    created_folders = []
    # Iterate through each class, copy images to new class subfolders  dataset.samples
    #for class_name, class_idx in dataset.label_map.items():
    # Iterate through each class, copy images to new class subfolders
    for class_name, class_idx in dataset.label_map.items():  # Iterate through label mapping
        # Create new directory path using mapped class index
        new_class_folder_path = os.path.join(new_train_dir, f"{class_idx:05d}")
        
        # If folder doesn't exist, create it
        if not os.path.exists(new_class_folder_path):
            os.makedirs(new_class_folder_path)
            created_folders.append(new_class_folder_path)  # Record newly created folder
        
        # Iterate through all samples in the dataset, copy images belonging to current class
        for img_path, label in dataset.samples:
            # Use mapped class index for matching
            mapped_class_idx = dataset.label_map.get(label, label)  # Mapped class index
            if mapped_class_idx == class_idx:  # If class index matches
                shutil.copy(img_path, new_class_folder_path)  # Copy image to new folder


    # Print transfer information
    print(f"Total of {len(created_folders)} new folders were created:")
    for folder in created_folders:
        print(f"Created folder: {folder}")

    print("\nImages have been successfully copied to the new class-specific folders.")

