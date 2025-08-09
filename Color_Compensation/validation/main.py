import os
import random
import warnings
import argparse
from argument import args  

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import LambdaLR
import math
import time
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from synthesize_utils import load_model
from utils import (
    ImageFolder,
    ShufflePatches,
    mix_aug,
    AverageMeter,
    accuracy,
    get_parameters,
)

# Need to modify the checkpoint path in ./synthesize_utils.py to load the teacher model from RDED paper



sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)

def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    main_worker(args)


def main_worker(args):
    print("=> using pytorch pre-trained teacher model '{}'".format(args.arch_name))
    teacher_model = load_model(
        model_name=args.arch_name,
        dataset=args.subset,
        pretrained=True,
        classes=args.classes,
    )

    student_model = load_model(
        model_name=args.stud_name,
        dataset=args.subset,
        pretrained=False,
        classes=args.classes,
    )
    teacher_model = torch.nn.DataParallel(teacher_model).cuda()
    student_model = torch.nn.DataParallel(student_model).cuda()

    teacher_model.eval()
    student_model.train()

    # freeze all layers
    for param in teacher_model.parameters():
        param.requires_grad = False

    cudnn.benchmark = True

    # optimizer
    if args.sgd:
        optimizer = torch.optim.SGD(
            get_parameters(student_model),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            get_parameters(student_model),
            lr=args.adamw_lr,
            betas=[0.9, 0.999],
            weight_decay=args.adamw_weight_decay,
        )

    # lr scheduler
    if args.cos == True:
        scheduler = LambdaLR(
            optimizer,
            lambda step: 0.5 * (1.0 + math.cos(math.pi * step / args.re_epochs / 2))
            if step <= args.re_epochs
            else 0,
            last_epoch=-1,
        )
    else:
        scheduler = LambdaLR(
            optimizer,
            lambda step: (1.0 - step / args.re_epochs) if step <= args.re_epochs else 0,
            last_epoch=-1,
        )

    print("process data from {}".format(args.syn_data_path))
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    augment = []
    augment.append(transforms.ToTensor())
    augment.append(ShufflePatches(args.factor))
    augment.append(
        transforms.RandomResizedCrop(
            size=args.input_size,
            scale=(1 / args.factor, args.max_scale_crops),
            antialias=True,
        )
    )
    augment.append(transforms.RandomHorizontalFlip())
    augment.append(normalize)

    train_dataset = ImageFolder(
        classes=range(args.nclass),
        ipc=args.ipc,
        mem=True,
        shuffle=True,
        root=args.syn_data_path,
        transform=transforms.Compose(augment),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.re_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=set_worker_sharing_strategy,
    )
    if args.subset in ['tinyimagenet', "imagenet-1k"]:
        args.re_batch_size = 2000
        val_dataset = datasets.ImageFolder(root=args.val_dir, transform=transforms.Compose(
                    [
                        transforms.Resize(args.input_size // 7 * 8, antialias=True),
                        transforms.CenterCrop(args.input_size),
                        transforms.ToTensor(),
                        normalize,
                    ]),
                )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.re_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            worker_init_fn=set_worker_sharing_strategy,
        )
    else:
        if args.subset in ["cifar10","cifar100"]:
            args.re_batch_size = 5000
        elif args.subset in ['imagenette',"imagenet-woof"]:
            args.re_batch_size = 50  
        val_loader = torch.utils.data.DataLoader(
            ImageFolder(
                classes=args.classes,
                ipc=args.val_ipc,
                mem=True,
                root=args.val_dir,
                transform=transforms.Compose(
                    [
                        transforms.Resize(args.input_size // 7 * 8, antialias=True),
                        transforms.CenterCrop(args.input_size),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            ),
            batch_size=args.re_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            worker_init_fn=set_worker_sharing_strategy,
        )
    print("load data successfully")

    best_acc1 = 0
    best_epoch = 0
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.train_loader = train_loader
    args.val_loader = val_loader

    for epoch in range(args.re_epochs):
        train(epoch, train_loader, teacher_model, student_model, args)

        if epoch % 10 == 9 or epoch == args.re_epochs - 1:
            if epoch > args.re_epochs * 0.01:
                top1 = validate(student_model, args, epoch)
                print("Best Acc Now:",best_acc1)
            else:
                top1 = 0
        else:
            top1 = 0
        
        scheduler.step()
        if top1 > best_acc1:
            best_acc1 = max(top1, best_acc1)
            best_epoch = epoch
            

    print(f"Train Finish! Best accuracy is {best_acc1}({best_epoch})")


def train(epoch, train_loader, teacher_model, student_model, args):
    """Train the model, with optional knowledge distillation."""
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    optimizer = args.optimizer
    loss_function_ce = nn.CrossEntropyLoss()
    loss_function_kl = nn.KLDivLoss(reduction="batchmean")

    if args.kd:
        teacher_model.eval()
    student_model.train()

    t1 = time.time()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        # Generate mixed images for augmentation
        mix_images, _, _, _ = mix_aug(images, args)

        pred_label = student_model(images)

        if args.kd:
            with torch.no_grad():
                # Generate soft labels using teacher model
                soft_mix_label = teacher_model(mix_images)
                soft_mix_label = F.softmax(soft_mix_label / args.temperature, dim=1)

            # Forward pass for student model with augmented images
            pred_mix_label = student_model(mix_images)

            # Compute soft predictions
            soft_pred_mix_label = F.log_softmax(pred_mix_label / args.temperature, dim=1)

            # Compute distillation loss
            loss_kd = loss_function_kl(soft_pred_mix_label, soft_mix_label)
        else:
            # Compute cross-entropy loss
            loss_ce = loss_function_ce(pred_label, labels)

        # Combine losses if KD is enabled
        if args.kd:
            loss = loss_kd 
        else:
            loss = loss_ce

        loss = loss / args.re_accum_steps

        # Backpropagation
        if batch_idx % args.re_accum_steps == 0:
            optimizer.zero_grad()
        loss.backward()
        if batch_idx % args.re_accum_steps == (args.re_accum_steps - 1):
            optimizer.step()

        # Update metrics
        prec1, prec5 = accuracy(pred_label, labels, topk=(1, 5))
        n = images.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

    # Print training information
    printInfo = (
        "TRAIN Iter {}: loss = {:.6f},\t".format(epoch, objs.avg)
        + "Top-1 acc = {:.6f},\t".format(top1.avg)
        + "Top-5 acc = {:.6f},\t".format(top5.avg)
        + "train_time = {:.6f}".format((time.time() - t1))
    )
    print(printInfo)
    t1 = time.time()

def validate(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for data, target in args.val_loader:
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    logInfo = (
        "TEST:\nIter {}: loss = {:.6f},\t".format(epoch, objs.avg)
        + "Top-1 acc = {:.6f},\t".format(top1.avg)
        + "Top-5 acc = {:.6f},\t".format(top5.avg)
        + "val_time = {:.6f}".format(time.time() - t1)
    )
    print(logInfo)
    logInfo = (
        "TEST: Iter {}: loss = {:.6f},\t".format(epoch, objs.avg)
        + "Top-1 acc = {:.6f},\t".format( top1.avg)
        + "Top-5 acc = {:.6f},\t".format(top5.avg)
        + "val_time = {:.6f}".format(time.time() - t1)
    )
    return top1.avg


if __name__ == "__main__":
    #pass
    main(args)