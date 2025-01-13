import os
import csv
import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
import lpips
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np

def calculate_metrics_for_folder(folder_path, output_file, device="cuda"):
    # 初始化指标计算器
    fid = FrechetInceptionDistance(feature=2048).to(device)
    inception_score = InceptionScore(feature=2048, num_splits=1).to(device)
    kid = KernelInceptionDistance(subset_size=30).to(device)
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    # 转换器
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8))
    ])

    # 图像张量
    real_tensors, fake_tensors = [], []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png"):
            file_path = os.path.join(folder_path, file_name)
            img = Image.open(file_path).convert("RGB")

            # 裁剪区域
            gt_box = (400, 140, 625, 365)
            gen_box = (675, 140, 900, 365)
            gt_img = img.crop(gt_box)
            gen_img = img.crop(gen_box)

            # 转换为 Tensor
            gt_tensor = transform(gt_img).to(device)
            gen_tensor = transform(gen_img).to(device)

            real_tensors.append(gt_tensor)
            fake_tensors.append(gen_tensor)

            # 更新 Inception Score
            inception_score.update(gen_tensor.unsqueeze(0))

    # 计算 FID 和 KID
    real_tensors = torch.stack(real_tensors)
    fake_tensors = torch.stack(fake_tensors)
    fid.update(real_tensors, real=True)
    fid.update(fake_tensors, real=False)
    fid_score = fid.compute().item()
    
    is_mean, is_std = inception_score.compute()

    kid.update(real_tensors, real=True)
    kid.update(fake_tensors, real=False)
    kid_mean, kid_std = kid.compute()

    # LPIPS 和 SSIM
    lpips_scores, ssim_scores = [], []
    for real_img, fake_img in zip(real_tensors, fake_tensors):
        lpips_score = lpips_fn(real_img.unsqueeze(0), fake_img.unsqueeze(0)).item()
        lpips_scores.append(lpips_score)

        real_np = real_img.permute(1, 2, 0).cpu().numpy()
        fake_np = fake_img.permute(1, 2, 0).cpu().numpy()
        ssim_score = compare_ssim(real_np, fake_np, multichannel=True)
        ssim_scores.append(ssim_score)

    lpips_mean = np.mean(lpips_scores)
    ssim_mean = np.mean(ssim_scores)

    # 结果保存
    new_results = [
        ["Results_Folder", folder_path],
        ["FID", fid_score],
        ["Inception Score Mean", is_mean.item()],
        ["Inception Score Std", is_std.item()],
        ["KID Mean", kid_mean.item()],
        ["KID Std", kid_std.item()],
        ["LPIPS", lpips_mean],
        ["SSIM", ssim_mean]
    ]

    # 确保不覆盖旧数据
    write_mode = "a" if os.path.exists(output_file) else "w"
    with open(output_file, mode=write_mode, newline="") as f:
        writer = csv.writer(f)
        writer.writerow([])
        for row in new_results:
            writer.writerow(row)

    print(f"Metrics saved to {output_file}")

# 调用函数
device = "cuda" if torch.cuda.is_available() else "cpu"
test_results_folder = "./results/ori_model/shoes/test_results_epoch1000"
output_csv_file = "metrics_results.csv"

calculate_metrics_for_folder(test_results_folder, output_csv_file, device=device)
