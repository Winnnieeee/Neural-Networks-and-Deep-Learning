import os
from PIL import Image

# 定义路径
input_base_dir = "BIPED"  # 原始数据集目录
output_base_dir = "BIPED_resized"  # 处理后的数据存储目录

# 图像处理目标大小
target_size = (256, 256)

# 数据集子文件夹结构
subfolders = [
    ("edges/edge_maps/test/rgbr", "edges_resized/test"),
    ("edges/edge_maps/train/rgbr/real", "edges_resized/train"),
    ("edges/imgs/test/rgbr", "imgs_resized/test"),
    ("edges/imgs/train/rgbr/real", "imgs_resized/train"),
]

def resize_and_save(input_dir, output_dir, size):
    """
    调整图像大小并保存到目标文件夹
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        if os.path.isfile(img_path):
            try:
                # 打开图像并调整大小
                with Image.open(img_path) as img:
                    img_resized = img.resize(size, Image.Resampling.LANCZOS)
                    # 保存到目标文件夹
                    output_path = os.path.join(output_dir, img_name)
                    img_resized.save(output_path)
                    print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

# 遍历每个子文件夹并处理
for subfolder, output_subfolder in subfolders:
    input_dir = os.path.join(input_base_dir, subfolder)
    output_dir = os.path.join(output_base_dir, output_subfolder)
    resize_and_save(input_dir, output_dir, target_size)

print("所有图像已处理并保存到目标文件夹。")
