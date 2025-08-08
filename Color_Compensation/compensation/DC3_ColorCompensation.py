import os
from torch.utils.data import Dataset
from PIL import Image
import random
from augment.utils import Utils


class DC3_ColorCompensation(Dataset):
    def __init__(self, original_dataset, num_images, guidance_scale, idx_to_class, model_handler, ipc, label_map, dataset_name, combine_mode):
        self.original_dataset = original_dataset
        self.idx_to_class = idx_to_class
        self.model_handler = model_handler
        self.num_compensated_images_per_image = num_images
        self.guidance_scale = guidance_scale
        self.utils = Utils()
        self.ipc = ipc  # Number of images to select
        self.label_map = label_map
        self.dataset_name = dataset_name
        self.combine_mode = combine_mode
        self.compensated_images = self.generate_compensated_images()

    def generate_compensated_images(self):
        compensated_data = []

        base_directory = './DC3/DC3_ColorCompensation/'
        original_resized_dir = os.path.join(base_directory, 'original_resized')
        generated_dir = os.path.join(base_directory, 'generated')
        concatenated_dir = os.path.join(base_directory, 'concatenated')
        compensated_dir = os.path.join(base_directory, f'compensated_{self.dataset_name}_{self.ipc}')
        prompt_nums = len(self.prompts)
        # Ensure these directories exist
        os.makedirs(original_resized_dir, exist_ok=True)
        os.makedirs(generated_dir, exist_ok=True)
        os.makedirs(concatenated_dir, exist_ok=True)
        os.makedirs(compensated_dir, exist_ok=True)
        
        class_samples = {}
        for img_path, label_idx in self.original_dataset.samples:
            if self.dataset_name in ["imagenette", "imagenet-woof"]:
                mapped_label_idx = self.label_map[label_idx]
                label = self.idx_to_class[mapped_label_idx] 
            else:
                label = self.idx_to_class[label_idx]

            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append((img_path, label_idx))

        
        selected_samples = []
        for label, samples in class_samples.items():
            if len(samples) < self.ipc:
                raise ValueError(f"Class {label} has fewer samples than specified ipc={self.ipc}, cannot proceed with selection")
            selected_samples += samples
        print("len(selected_samples)",len(selected_samples))

        for idx, (img_path, label_idx) in enumerate(selected_samples):
            if self.dataset_name in ["imagenette", "imagenet-woof"]:
                mapped_label_idx = self.label_map[label_idx]
                label = self.idx_to_class[mapped_label_idx] 
            else:
                label = self.idx_to_class[label_idx]

            original_img = Image.open(img_path).convert('RGB')
            original_img = original_img.resize((256, 256))
            img_filename = os.path.basename(img_path)

            label_dirs = {dtype: os.path.join(base_directory, dtype, str(label)) for dtype in
                          ['original_resized', 'generated', 'concatenated', 'compensated']}

            label_dirs['compensated'] = os.path.join(base_directory, f'compensated_{self.combine_mode}_{prompt_nums}_{self.dataset_name}_{self.ipc}', str(label))

            for dir_path in label_dirs.values():
                os.makedirs(dir_path, exist_ok=True)

            original_img.save(os.path.join(label_dirs['original_resized'], img_filename))

            
            stepth = 2
            warm_prompt = ["sunny", "golden hour", "sepia", "vivid colors","daylight"]
            cold_prompt = ["rainy", "snowy", "frozen lake", "infrared","underwater"]
            selected_warm_prompt = random.sample(warm_prompt, 1)
            selected_cold_prompt = random.sample(cold_prompt, 1)

            

            compensated_images = []
            for prompt in selected_prompts:
                compensated_images += self.model_handler.generate_images(prompt, img_path, self.num_compensated_images_per_image, self.guidance_scale)

            # Concatenate every two generated images

            for i in range(0, len(compensated_images), stepth):

                img1 = compensated_images[i].resize((256, 256))
                img2 = compensated_images[i + 1].resize((256, 256))
                # Concatenate these two generated images
                combined_img = self.utils.combine_images(img1, img2, self.combine_mode)


                compensated_img = combined_img

                compensated_img_filename = f"{img_filename[:-3]}_compensated_{selected_prompts[0]}_{selected_prompts[1]}_{idx}.jpg"
                #print('Idx,STEPTH',idx,stepth)
                compensated_img.save(os.path.join(label_dirs['compensated'], compensated_img_filename))

                compensated_data.append((compensated_img, label))

        return compensated_data

    def __len__(self):
        return len(self.compensated_images)

    def __getitem__(self, idx):
        image, label = self.compensated_images[idx]
        return image, label
