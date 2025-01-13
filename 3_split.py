from PIL import Image
import os

def split_image(input_dir, output_dir_edges, output_dir_imgs, size=256):
    # 确保输出目录存在
    os.makedirs(output_dir_edges, exist_ok=True)
    os.makedirs(output_dir_imgs, exist_ok=True)
    
    # 遍历输入目录中的所有图像文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            # 构建完整的文件路径
            file_path = os.path.join(input_dir, filename)
            # 打开图像
            with Image.open(file_path) as img:
                # 确保图像宽度为512，高度为256
                if img.size == (512, 256):
                    # 切割图像为两个256x256的图像
                    left_img = img.crop((0, 0, size, size))
                    right_img = img.crop((size, 0, 2 * size, size))
                    
                    # 保存左边的图像到edges目录
                    left_output_path = os.path.join(output_dir_edges, filename)
                    left_img.save(left_output_path)
                    
                    # 保存右边的图像到imgs目录
                    right_output_path = os.path.join(output_dir_imgs, filename)
                    right_img.save(right_output_path)
                else:
                    print(f"Image {filename} does not have the expected size (512x256).")

# 输入目录
input_dir = './Shoes/train'
# 输出目录
output_dir_edges = './SHOES_resized/edges_resized/train'
output_dir_imgs = './SHOES_resized/imgs_resized/train'

# 开始切割图像
split_image(input_dir, output_dir_edges, output_dir_imgs)

input_dir = './Shoes/test'
output_dir_edges = './SHOES_resized/edges_resized/test'
output_dir_imgs = './SHOES_resized/imgs_resized/test'
split_image(input_dir, output_dir_edges, output_dir_imgs)