import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityd,
    CropForegroundd,
    Resized,
)
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism

# 1. 设置随机种子，保证可复现
set_determinism(seed=0)

# 2. 定义数据路径
data_dir = os.path.expanduser("~/projects/Hydro_Seg_Project/data")
print(f"Data Directory: {data_dir}")

# 查找所有 sub_XXX 文件夹
subjects = sorted(glob.glob(os.path.join(data_dir, "sub_*")))
print(f"Found {len(subjects)} subjects.")

# 3. 构建数据字典列表 (Image + Label)
data_dicts = []
for sub in subjects:
    # 假设每个 sub 下面只有一个 scan_XX，或者我们只取第一个
    scan_dirs = sorted(glob.glob(os.path.join(sub, "scan_*")))
    if len(scan_dirs) == 0:
        continue
    
    scan_dir = scan_dirs[0] # 取第一个 scan
    
    img_path = os.path.join(scan_dir, "eT1W_FFE_SVR.nii.gz")
    # 注意：这里我们使用 gt.nii.gz (金标准)
    lbl_path = os.path.join(scan_dir, "segmentation", "gt.nii.gz")
    
    if os.path.exists(img_path) and os.path.exists(lbl_path):
        data_dicts.append({"image": img_path, "label": lbl_path})
    else:
        print(f"Warning: Missing files in {scan_dir}")

print(f"Valid datasets: {len(data_dicts)}")
# 取少量数据做测试
train_files = data_dicts[:3] 

# 4. 定义 MONAI 变换 (Transforms)
train_transforms = Compose(
    [
        # 加载图像和标签
        LoadImaged(keys=["image", "label"]),
        # 确保通道在前: (Channel, H, W, D)
        EnsureChannelFirstd(keys=["image", "label"]),
        # 统一方向为 RAS (Right, Anterior, Superior)
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # 统一体素间距为 1mm (虽然原始数据已经是1mm，但为了保险)
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        # 强度归一化 (只针对图像，标签不能归一化)
        ScaleIntensityd(keys=["image"]),
        # (可选) 裁剪掉周围的空气背景，减少计算量
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # 调整大小以便快速测试 (训练时可以调大或使用随机裁剪)
        Resized(keys=["image", "label"], spatial_size=(96, 96, 96), mode=("trilinear", "nearest")),
    ]
)

# 5. 创建 DataLoader
check_ds = Dataset(data=train_files, transform=train_transforms)
check_loader = DataLoader(check_ds, batch_size=1, num_workers=2)

# 6. 检查一个 Batch 的数据
def check_loader_output():
    print("-" * 30)
    print("Checking DataLoader output...")
    
    # 获取第一个 batch
    first_batch = next(iter(check_loader))
    images = first_batch["image"]
    labels = first_batch["label"]
    
    print(f"Image Batch Shape: {images.shape}") # 应该是 (1, 1, 96, 96, 96)
    print(f"Label Batch Shape: {labels.shape}") # 应该是 (1, 1, 96, 96, 96)
    
    # 检查标签的唯一值，确保 Resize 没有破坏整数标签
    unique_lbls = torch.unique(labels)
    print(f"Unique Labels in Batch: {unique_lbls}")
    
    # 简单可视化 (保存切片)
    slice_idx = 48 # 中间切片
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Image (Normalized)")
    plt.imshow(images[0, 0, :, :, slice_idx], cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title("Label (GT)")
    plt.imshow(labels[0, 0, :, :, slice_idx], cmap="jet", interpolation="nearest")
    plt.axis("off")
    
    plt.savefig("loader_check.png")
    print("Saved preview to 'loader_check.png'")

if __name__ == "__main__":
    check_loader_output()