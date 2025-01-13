from PIL import Image
import os

def invert_colors(image_path, output_path):
    # 打开图像
    with Image.open(image_path) as img:
        # 转换图像颜色模式为'L'（灰度），便于处理
        img = img.convert('L')
        # 反转像素值：黑底白边 -> 白底黑边
        inverted_img = Image.eval(img, lambda x: 255 - x)
        # 保存反转后的图像
        inverted_img.save(output_path)

def process_images_in_folder(input_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历输入目录中的所有文件和文件夹
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                # 构建完整的文件路径
                file_path = os.path.join(root, file)
                # 构建输出目录的相应路径
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_file_path = os.path.join(output_subdir, file)
                
                # 处理图像并保存到输出目录
                invert_colors(file_path, output_file_path)
                print(f"Processed {file_path} -> {output_file_path}")

# 输入和输出目录
input_dir = './BIPED_resized/edges_resized'
output_dir = './BIPED_resized_BW/edge_resized'

# 开始处理图像
process_images_in_folder(input_dir, output_dir)