import os
import time
import torch
import argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from util.utils import str_to_bool
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

import dq.methods as methods
import dq.datasets as datasets
from diffusers import StableDiffusionPipeline 
import torch.nn as nn
def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # Basic arguments
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ResNet18', help='model')
    parser.add_argument('--batch', type=int, default=128, help='the number of batch size for selection')
    parser.add_argument('--gpu', default=None, nargs="+", type=int, help='gpu id to use')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--print_freq', '-p', default=50, type=int, help='print frequency (default: 20)')
    parser.add_argument('--seed', default=int(time.time() * 1000) % 100000, type=int, help="random seed")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--K', type=int, default=10, help='image per class')
    # Selecting
    parser.add_argument('--balance', default=True, type=str_to_bool, help="whether balance selection is performed per class")
    parser.add_argument('--replace', action='store_true', default=False, help='whether the samples can be selected repeatedly')

    # Checkpoint and resumption
    parser.add_argument('--save_path', "-sp", type=str, default='', help='path to save results (default: do not save)')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.selection_batch = args.batch
    if args.save_path != "" and not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    # No longer need the experiment loop
    print("Starting dataset selection...")

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[args.dataset](
        args.data_path)
    args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names

    torch.random.manual_seed(args.seed)

    # Initialize the available indices
    avail_indices = np.arange(len(dst_train))

    # Re-initialize the training set with the remaining indices
    dst_train = torch.utils.data.Subset(dst_train, avail_indices)
    print('Dst Size: {}'.format(len(dst_train)))

    # Get embeddings from model and apply DBSCAN clustering
    
    dst_train_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch, shuffle=False)
    diffusion_checkpoints_path = "/data/stablediffusion/checkpoints/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(diffusion_checkpoints_path, torch_dtype=torch.float16)
    pipe = pipe.to(args.device)
    pipe = pipe.to(torch.float16)

    latents = []
    labels = []
    indices = []
    for input, label, indice in tqdm(dst_train_loader, desc="Processing batches", total=len(dst_train_loader)):
        labels.append(label.numpy())
        indices.append(indice.numpy())
        input = input.to(args.device).half()
        input = torch.nn.functional.interpolate(input, size=(512, 512))
        with torch.no_grad():
            # Pass the image through the diffusion model's encoder to get latents
            latent = pipe.vae.encode(input).latent_dist.mean  # latent_dist 0.18215  # Assume `encode` is the function that outputs latents  
            latents.append(latent.cpu().numpy())  # Collect latents from all batches

    print("latents has been prepared")
    latents = np.concatenate(latents, axis=0)
    latents = latents.reshape(latents.shape[0], -1) 
    print(latents.shape)

    # pca = PCA(n_components=50)
    # c_latents = pca.fit_transform(c_latents)
    # #del latents
    # print(c_latents.shape)
    
    labels = np.concatenate(labels, axis=0)
    indices = np.concatenate(indices, axis=0)

    dataset_indices = {}
    cluster_result = {}

    for c in range(args.num_classes):
        dataset_indices[c] = []
        cluster_result[c] = []

    for c in range(args.num_classes):
        c_mask = labels == c
        c_latents = latents[c_mask]
        pca = PCA(n_components=50)
        c_latents = pca.fit_transform(c_latents)

        c_mask = np.array(c_mask, dtype=bool)  
        dataset_indices[c].append(indices[c_mask])
        
        n_clusters = args.K
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # KMeans clustering
        kmeans.fit(c_latents)
        cluster_labels = kmeans.labels_
        cluster_result[c].append(cluster_labels)
        print(f"Classes {c}, len(labels): {len(cluster_labels)}, labels[:20]: {cluster_labels[:20]}")

        for cluster_id in range(n_clusters):
            target_indices = indices[c_mask][cluster_labels == cluster_id]
            file_name = f'class_{c}_cluster_{cluster_id}.npy'
            file_path = os.path.join(args.save_path, file_name)
            np.save(file_path, target_indices)
            print(f"Saved: {file_path}")

     # Get the number of clusters (excluding noise)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Number of clusters (excluding noise): {num_clusters}")



if __name__ == '__main__':
    main()
