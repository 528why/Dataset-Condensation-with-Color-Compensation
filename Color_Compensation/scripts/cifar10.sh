python3 main.py --train_dir "/data/DC3/cifar10/train_by_class" \
--dataset cifar10 \
--ipc 10 \
--combine_mode gradient \
--indices_path "./DC3/Submodular_Sampling/submodular_sampler/cifar10_k_010_ipc_10/sample.npy" \


python get_imagenet_by_class.py --dataset train  --ipc 10 --subset cifar10  --combine_mode gradient --prompt_nums 10 

cd ./validation

python main.py \
--subset "cifar10" \
--arch-name "resnet18_modified" \
--factor 2 \
--num-crop 5 \
--mipc 300 \
--ipc 10 \
--stud-name "resnet18_modified" \
--re-epochs 2000 \
--syn_data_path "/data/DC3/cifar10_gradient/ipc10_train_by_class"  \
--val_dir "/data/cifar10/validation_by_class" 