import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import csv
import json

DATASET = "BIPED_resized_BW" # BIPED_resized 或 BIPED_resized_BW 或 SHOES_resized

# 数据准备
train_edge_path = f"./dataset/{DATASET}/edges_resized/train/*.png"
train_img_path = f"./dataset/{DATASET}/imgs_resized/train/*.jpg"
test_edge_path = f"./dataset/{DATASET}/edges_resized/test/*.png"
test_img_path = f"./dataset/{DATASET}/imgs_resized/test/*.jpg"

train_edges = sorted(glob.glob(train_edge_path))
train_imgs = sorted(glob.glob(train_img_path))
test_edges = sorted(glob.glob(test_edge_path))
test_imgs = sorted(glob.glob(test_img_path))


print(f"训练集: 输入 {len(train_edges)}, 目标 {len(train_imgs)}")
print(f"测试集: 输入 {len(test_edges)}, 目标 {len(test_imgs)}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 定义数据集
class Pix2PixDataset(Dataset):
    def __init__(self, edge_paths, img_paths, transform):
        self.edge_paths = edge_paths
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        edge = Image.open(self.edge_paths[index]).convert("RGB")
        img = Image.open(self.img_paths[index]).convert("RGB")
        return self.transform(edge), self.transform(img)

    def __len__(self):
        return len(self.edge_paths)

train_dataset = Pix2PixDataset(train_edges, train_imgs, transform)
test_dataset = Pix2PixDataset(test_edges, test_imgs, transform)

BATCH_SIZE = 4
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 下采样模块
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.model(x)

# 上采样模块
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.model(x)

# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down1 = Downsample(3, 64)
        self.down2 = Downsample(64, 128)
        self.down3 = Downsample(128, 256)
        self.down4 = Downsample(256, 512)
        self.down5 = Downsample(512, 512)
        self.down6 = Downsample(512, 512)

        self.up1 = Upsample(512, 512)
        self.up2 = Upsample(1024, 512)
        self.up3 = Upsample(1024, 256)
        self.up4 = Upsample(512, 128)
        self.up5 = Upsample(256, 64)

        self.last = nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        u1 = self.up1(d6)
        u2 = self.up2(torch.cat([u1, d5], dim=1))
        u3 = self.up3(torch.cat([u2, d4], dim=1))
        u4 = self.up4(torch.cat([u3, d3], dim=1))
        u5 = self.up5(torch.cat([u4, d2], dim=1))

        return torch.tanh(self.last(torch.cat([u5, d1], dim=1)))

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            Downsample(6, 64),
            Downsample(64, 128),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 1, kernel_size=3)
        )

    def forward(self, edge, img):
        x = torch.cat([edge, img], dim=1)
        return torch.sigmoid(self.model(x))

# 初始化模型与优化器
device = "cuda" if torch.cuda.is_available() else "cpu"
gen = Generator().to(device)
dis = Discriminator().to(device)

g_optimizer = torch.optim.Adam(gen.parameters(), lr=0.001, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(dis.parameters(), lr=0.001, betas=(0.5, 0.999))
# g_optimizer = torch.optim.RMSprop(gen.parameters(), lr=0.001)
# d_optimizer = torch.optim.RMSprop(dis.parameters(), lr=0.001)

loss_fn = nn.BCELoss()
L1_loss_fn = nn.L1Loss()

# 训练模型
EPOCHS = 200
LAMBDA = 10
checkpoint_dir = "./checkpoints"
log_csv = "loss_log.csv"
os.makedirs(checkpoint_dir, exist_ok=True)

# 用于保存所有训练历史的字典
training_history = {
    "epochs": [],
    "d_loss": [],
    "g_loss": []
}

# 如果存在之前的训练历史，加载它
history_file = "training_history.json"
if os.path.exists(history_file):
    with open(history_file, 'r') as f:
        training_history = json.load(f)

# 检查并加载 checkpoint
start_epoch = 0
if os.path.exists(os.path.join(checkpoint_dir, "training_checkpoint.pth")):
    checkpoint = torch.load(os.path.join(checkpoint_dir, "training_checkpoint.pth"))
    gen.load_state_dict(checkpoint["gen"])
    dis.load_state_dict(checkpoint["dis"])
    g_optimizer.load_state_dict(checkpoint["g_optimizer"])
    d_optimizer.load_state_dict(checkpoint["d_optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resumed training from epoch {start_epoch + 1}")

# 如果 CSV 文件不存在，创建它并写入表头
if not os.path.exists(log_csv):
    with open(log_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "D Loss", "G Loss"])

# 开始训练
for epoch in range(start_epoch, EPOCHS):
    epoch_d_loss = 0
    epoch_g_loss = 0
    batch_count = 0
    
    for edge, img in train_loader:
        edge, img = edge.to(device), img.to(device)

        # 训练判别器
        dis.train()
        gen.eval()
        d_optimizer.zero_grad()
        real_output = dis(edge, img)
        fake_img = gen(edge).detach()
        fake_output = dis(edge, fake_img)
        d_loss = loss_fn(real_output, torch.ones_like(real_output, device=device)) + \
                 loss_fn(fake_output, torch.zeros_like(fake_output, device=device))
        d_loss.backward()
        d_optimizer.step()

        # 训练生成器
        gen.train()
        dis.eval()
        g_optimizer.zero_grad()
        fake_img = gen(edge)
        fake_output = dis(edge, fake_img)
        g_loss = loss_fn(fake_output, torch.ones_like(fake_output, device=device)) + \
                 LAMBDA * L1_loss_fn(fake_img, img)
        g_loss.backward()
        g_optimizer.step()
        
        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()
        batch_count += 1

    # 计算平均损失
    avg_d_loss = epoch_d_loss / batch_count
    avg_g_loss = epoch_g_loss / batch_count

    # 更新训练历史
    training_history["epochs"].append(epoch + 1)
    training_history["d_loss"].append(avg_d_loss)
    training_history["g_loss"].append(avg_g_loss)

    # 保存训练历史到JSON文件
    with open(history_file, 'w') as f:
        json.dump(training_history, f)

    # 追加到CSV文件
    with open(log_csv, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, avg_d_loss, avg_g_loss])

    # 保存单一checkpoint文件
    torch.save({
        "epoch": epoch,
        "gen": gen.state_dict(),
        "dis": dis.state_dict(),
        "g_optimizer": g_optimizer.state_dict(),
        "d_optimizer": d_optimizer.state_dict(),
        "training_history": training_history
    }, os.path.join(checkpoint_dir, "training_checkpoint.pth"))

    print(f"Epoch [{epoch + 1}/{EPOCHS}], D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")

# 绘制完整的loss曲线
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(training_history["epochs"], training_history["d_loss"], label="D Loss", color="pink")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Discriminator Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(training_history["epochs"], training_history["g_loss"], label="G Loss", color="purple")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Generator Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curves.png")
plt.close()

# 测试与保存可视化结果
os.makedirs("test_results", exist_ok=True)
with torch.no_grad():
    gen.eval()
    for idx, (edge, img) in enumerate(test_loader):
        edge, img = edge.to(device), img.to(device)
        fake_img = gen(edge)
        
        # 只保存指定数量的测试结果
        max_test_samples = 10  # 可以根据需要调整这个数字
        if idx >= max_test_samples:
            break

        # 遍历每个批次中的图像
        for i in range(edge.size(0)):
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 3, 1)
            plt.title("Input")
            plt.imshow((edge[i].permute(1, 2, 0).cpu().numpy() + 1) / 2)
            
            plt.subplot(1, 3, 2)
            plt.title("Ground Truth")
            plt.imshow((img[i].permute(1, 2, 0).cpu().numpy() + 1) / 2)
            
            plt.subplot(1, 3, 3)
            plt.title("Generated")
            plt.imshow((fake_img[i].permute(1, 2, 0).cpu().numpy() + 1) / 2)
            
            plt.savefig(f"test_results/result_{idx * edge.size(0) + i}.png")
            plt.close()