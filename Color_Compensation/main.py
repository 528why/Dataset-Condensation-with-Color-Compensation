import argparse
from torchvision import datasets
from compensation.handler import ModelHandler
from compensation.utils import Utils
from compensation.DC3_ColorCompensation import DC3_ColorCompensation
from utils import get_dataset
from torch.nn import DataParallel

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate an compensated dataset from original images.")
    parser.add_argument('--train_dir', type=str, required=True, help='Path to the directory containing the original training images.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name.')
    parser.add_argument('--ipc', type=int, required=True, help='ipc')
    parser.add_argument('--combine_mode', type=str, required=True, choices=['gradient', 'random', 'grid','fourfold_view'],
                        help='image combine_mode, choose either "gradient" or "random"')
    parser.add_argument('--indices_path', type=str, help='Path to the directory containing the indices path')
    return parser.parse_args()

def main():

    args = parse_arguments()
    prompts = args.prompts.split(',')  # This will give you a list of prompts

    # Initialize the model
    model_id = "/data/modelscope/hub/AI-ModelScope/instruct-pix2pix"
    model_initialization = ModelHandler(model_id=model_id, device='cuda')

    train_dataset, idx_to_class, label_map = get_dataset(args)

    # Create the compensatied dataset
    compensated_train_dataset = DC3_ColorCompensation(
        original_dataset=train_dataset,
        num_images=1,
        guidance_scale=4,
        idx_to_class = idx_to_class,
        model_handler=model_initialization,
        ipc=args.ipc,
        label_map=label_map,
        dataset_name=args.dataset,
        combine_mode = args.combine_mode,
    )
    print("get compensated_train_dataset:",len(compensated_train_dataset))

if __name__ == '__main__':
    main()

