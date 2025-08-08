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
    
# 设置参数解析器
parser = argparse.ArgumentParser(description='Process ImageNet-subset dataset.')
parser.add_argument('--dataset', choices=['train', 'validation'], required=True, help='Specify whether to process train or validation dataset')
parser.add_argument('--ipc', type=int, required=True, help='ipc')
parser.add_argument('--subset', type=str, required=True, help='datset name')
parser.add_argument('--combine_mode', type=str, required=True, choices=['gradient', 'random', 'grid','fourfold_view'],
                        help='image combine_mode, choose either "gradient" or "random"')
parser.add_argument('--prompt_nums', type=int, required=True, help='prompt_nums')
args = parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self, samples, transform=None, label_map=None):
        self.samples = samples
        self.transform = transform
        self.label_map = label_map
        # 如果有label_map, 使用该映射, 否则使用原始标签
        if label_map:
            self.classes = [str(label_map[key]) for key in sorted(label_map.keys())]
        else:
            original_labels = set([sample[1] for sample in samples])
            self.classes = [str(label) for label in sorted(original_labels)]
            
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.transform(Image.open(path).convert('RGB')) if self.transform else Image.open(path).convert('RGB')
        target = int(target)
        target = self.label_map.get(target, target)  # 如果有映射，替换为新标签
        return sample, target

    def __len__(self):
        return len(self.samples)

# 设置原始数据集路径和新文件夹路径
if args.dataset == 'train':
    if args.fractal:
        original_train_dir = f'/data/DC3/compensated_{args.combine_mode}_{args.prompt_nums}_{args.subset}_{args.ipc}'
        new_train_dir = f'/data/DC3/{args.subset}_{args.combine_mode}_{args.prompt_nums}/ipc{args.ipc}_train_by_class'
    else:
        original_train_dir = f'/data/DC3/compensated_{args.combine_mode}_{args.prompt_nums}_{args.subset}_{args.ipc}'
        new_train_dir = f'/data/DC3/{args.subset}_{args.combine_mode}_{args.prompt_nums}/ipc{args.ipc}_train_by_class'


# 确保新文件夹存在，如果不存在则创建
if not os.path.exists(new_train_dir):
    os.makedirs(new_train_dir)
if args.dataset == "train":
    # 使用 ImageFolder 加载数据集
    dataset = datasets.ImageFolder(root=original_train_dir if args.dataset == 'train' else original_train_dir, transform=None)

    # 获取标签到类别的映射（ImageFolder 自动处理了类别索引）
    class_to_idx = dataset.class_to_idx

    # 遍历每个类别，将图像复制到新的类别子文件夹
    for class_name, class_idx in class_to_idx.items():
        # 创建新目录路径
        new_class_folder_path = os.path.join(new_train_dir, f"{class_idx:05d}")
        if not os.path.exists(new_class_folder_path):
            os.makedirs(new_class_folder_path)
        
        # 遍历数据集中的所有样本，复制属于当前类别的图像
        for img_path, label in dataset.samples:
            if label == class_idx:  # 如果标签匹配
                shutil.copy(img_path, new_class_folder_path)

if args.dataset == 'validation':
    # 使用 ImageFolder 加载数据集
    dataset = datasets.ImageFolder(root=original_train_dir, transform=None)
    class_to_idx = dataset.class_to_idx
    if args.subset == "imagenette":
    # 只选择指定的10个类别
        sub_imagenet_indices = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
    elif args.subset == "imagenet-woof":
        sub_imagenet_indices = [193, 182, 258, 162, 155, 167, 159, 273, 207, 229]
    # 筛选出属于这10个类别的样本
    imagenet_filtered = [sample for sample in dataset.samples if sample[1] in sub_imagenet_indices]
    # 更新标签映射
    idx_to_class = {i: dataset.classes[sub_imagenet_indices[i]] for i in range(len(sub_imagenet_indices))}
    label_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(sub_imagenet_indices)}
    # 使用自定义标签映射创建validation集
    dataset = CustomDataset(imagenet_filtered, transform=None, label_map=label_map)
    
    # 汇报创建的新文件夹
    created_folders = []
    # 遍历每个类别，将图像复制到新的类别子文件夹  dataset.samples
    #for class_name, class_idx in dataset.label_map.items():
    # 遍历每个类别，将图像复制到新的类别子文件夹
    for class_name, class_idx in dataset.label_map.items():  # 遍历标签映射
        # 创建新目录路径，使用映射后的类别索引
        new_class_folder_path = os.path.join(new_train_dir, f"{class_idx:05d}")
        
        # 如果文件夹不存在，创建它
        if not os.path.exists(new_class_folder_path):
            os.makedirs(new_class_folder_path)
            created_folders.append(new_class_folder_path)  # 记录新建的文件夹
        
        # 遍历数据集中的所有样本，复制属于当前类别的图像
        for img_path, label in dataset.samples:
            # 使用映射后的类别索引来匹配
            mapped_class_idx = dataset.label_map.get(label, label)  # 映射后的类别索引
            if mapped_class_idx == class_idx:  # 如果类别索引匹配
                shutil.copy(img_path, new_class_folder_path)  # 复制图片到新文件夹


    # 打印转移的信息
    print(f"Total of {len(created_folders)} new folders were created:")
    for folder in created_folders:
        print(f"Created folder: {folder}")

    print("\nImages have been successfully copied to the new class-specific folders.")

