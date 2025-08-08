python get_bins.py --dataset CIFAR10 --model ResNet18  --batch 30 --gpu 0 --data_path /data/cifar10  --save_path ./kmeans/cluster_cifar10_k_010 --seed 42 --K 10

python -u submodular_sampling.py  --IPC  10 --K 10 --dataset CIFAR10 --data_path /data/cifar10     --num_exp 10 --workers 10 -se 0 --selection Submodular_sampler --model ResNet18     -sp ./submodular_sampler/cifar10_k_010_ipc_10     --batch 128 --submodular GraphCut --submodular_greedy NaiveGreedy --cluster_path  ./DC3/Submodular_Sampling/kmeans/cluster_cifar10_k_010   --pretrained    
