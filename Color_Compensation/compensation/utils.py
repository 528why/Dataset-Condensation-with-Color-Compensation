import os
import random
import numpy as np
from PIL import Image

class Utils:

    @staticmethod
    def load_fractal_images(fractal_img_dir):
        fractal_img_paths = [os.path.join(fractal_img_dir, fname) for fname in os.listdir(fractal_img_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]
        return [Image.open(path).convert('RGB').resize((256, 256)) for path in fractal_img_paths]

    @staticmethod
    def blend_images_with_resize(base_img, overlay_img, alpha=0.20):
        overlay_img_resized = overlay_img.resize(base_img.size)
        base_array = np.array(base_img, dtype=np.float32)
        overlay_array = np.array(overlay_img_resized, dtype=np.float32)
        assert base_array.shape == overlay_array.shape and len(base_array.shape) == 3
        blended_array = (1 - alpha) * base_array + alpha * overlay_array
        blended_array = np.clip(blended_array, 0, 255).astype(np.uint8)
        blended_img = Image.fromarray(blended_array)
        return blended_img

    @staticmethod
    def combine_images(original_img, augmented_img, combine_mode, blend_width=20):
        width, height = original_img.size
        if combine_mode == 'gradient':
            # 现有的渐变组合方式
            combine_choice = random.choice(['horizontal', 'vertical'])

            if combine_choice == 'vertical':  # Vertical combination
                mask = np.linspace(0, 1, blend_width).reshape(-1, 1)
                mask = np.tile(mask, (1, width))  # Extend mask horizontally
                mask = np.vstack([np.zeros((height // 2 - blend_width // 2, width)), mask,
                                  np.ones((height // 2 - blend_width // 2 + blend_width % 2, width))])
                mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))

            else:
                mask = np.linspace(0, 1, blend_width).reshape(1, -1)
                mask = np.tile(mask, (height, 1))  # Extend mask vertically
                mask = np.hstack([np.zeros((height, width // 2 - blend_width // 2)), mask,
                                  np.ones((height, width // 2 - blend_width // 2 + blend_width % 2))])
                mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))

        elif combine_mode == 'random':  # 新增的随机组合方式
            mask = np.random.rand(height, width, 3)  # 创建一个与图像尺寸相同的随机掩码
            mask = (mask < 0.5).astype(np.float32)  # 50%比例为原图，50%为增强图
        elif combine_mode == 'grid':  # 新增的网格组合方式
            grid_size = 8  # 16x16的方格
            mask = np.zeros((height, width, 3), dtype=np.float32)
            for i in range(0, height, grid_size):
                for j in range(0, width, grid_size):
                    # 随机决定当前方格由哪个图像来填充
                    if random.random() < 0.5:
                        mask[i:i + grid_size, j:j + grid_size, :] = 1
        else:
            raise ValueError(f"Unknown combine_mode: {combine_mode}")

        original_array = np.array(original_img, dtype=np.float32) / 255.0
        augmented_array = np.array(augmented_img, dtype=np.float32) / 255.0

        # 根据mask决定组合方式
        blended_array = (1 - mask) * original_array + mask * augmented_array
        blended_array = np.clip(blended_array * 255, 0, 255).astype(np.uint8)

        blended_img = Image.fromarray(blended_array)
        return blended_img
    
    @staticmethod
    def combine_images_4(augmented_img1, augmented_img2, augmented_img3, augmented_img4, blend_width=20):
        width, height = augmented_img1.size
    
        # 确保四张图像的大小一致
        assert augmented_img2.size == augmented_img3.size == augmented_img4.size == augmented_img1.size, "All input images must have the same size."
        
        # 创建一个新的输出图像，大小与原图相同
        out_image = Image.new("RGB", (width, height))
        
        # 将四张图像分别缩放至 1/2 大小，并拼接成一个 2x2 的网格
        img1_resized = augmented_img1.resize((width // 2, height // 2))
        img2_resized = augmented_img2.resize((width // 2, height // 2))
        img3_resized = augmented_img3.resize((width // 2, height // 2))
        img4_resized = augmented_img4.resize((width // 2, height // 2))
        
        # 将四张缩放后的图像分别粘贴到输出图像的相应位置
        out_image.paste(img1_resized, (0, 0))  # 左上
        out_image.paste(img2_resized, (0, height // 2))  # 左下
        out_image.paste(img3_resized, (width // 2, 0))  # 右上
        out_image.paste(img4_resized, (width // 2, height // 2))  # 右下

        return out_image 
      

    @staticmethod
    def is_black_image(image):
        histogram = image.convert("L").histogram()
        return histogram[-1] > 0.9 * image.size[0] * image.size[1] and max(histogram[:-1]) < 0.1 * image.size[0] * \
            image.size[1]