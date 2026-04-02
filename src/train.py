import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirFstd,
    Orientationd,
    Spacingd,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

# --- 1. 配置参数 (Configuration) ---
# 定义数据路径
data_dir = os.path.expanduser("~/projects/Hydro_Seg_Project/data")
# 模型保存路径
log_dir = os.path.expanduser("~/projects/Hydro_Seg_Project/experiments")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 训练超参数
max_epochs = 100          # 训练轮数 (如果是测试跑通可以改小，比如 5)
val_interval = 2          # 每隔多少个 epoch 做一次验证
batch_size = 2            # 3090显存很大，可以尝试 2 或 4
lr = 1e-4                 # 学习率
roi_size = (96, 96, 96)   # 训练时的裁剪尺寸
num_classes = 6           # 0-5 共6类

# 设置随机种子保证可复现
set_determinism(seed=0)

# --- 2. 数据准备 (Data Preparation) ---
print(f"Searching for data in: {data_dir}")
subjects = sorted(glob.glob(os.path.join(data_dir, "sub_*")))
data_dicts = []

for sub in subjects:
    # 尝试寻找 scan_01, 如果没有就找 scan_02 (为了尽可能多找数据)
    scan_dirs = sorted(glob.glob(os.path.join(sub, "scan_*")))
    if len(scan_dirs) == 0: continue
    
    # 优先用 scan_01，如果没有就用列表里的第一个
    scan_path = scan_dirs[0] 
    
    img_path = os.path.join(scan_path, "eT1W_FFE_SVR.nii.gz")
    lbl_path = os.path.join(scan_path, "segmentation", "gt.nii.gz")
    
    if os.path.exists(img_path) and os.path.exists(lbl_path):
        data_dicts.append({"image": img_path, "label": lbl_path})

print(f"Total valid subjects found: {len(data_dicts)}")

# 划分训练集和验证集 (80% 训练, 20% 验证)
# 43个样本 -> 约34个训练, 9个验证
train_files, val_files = data_dicts[:-9], data_dicts[-9:]
print(f"Training samples: {len(train_files)}, Validation samples: {len(val_files)}")

# --- 3. 数据变换 (Transforms) ---
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityd(keys=["image"]),
        # 关键：从大图中随机裁剪出 96x96x96 的块进行训练
        # label_key="label" 保证我们更有可能裁剪到有前景(非背景)的区域
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            pos=1,
            neg=1,
            num_samples=2, # 每个原图裁出2个块
            image_key="image",
            image_threshold=0,
        ),
        # 数据增强：随机翻转和旋转
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        EnsureTyped(keys=["image", "label"]),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityd(keys=["image"]),
        EnsureTyped(keys=["image", "label"]),
    ]
)

# --- 4. DataLoader ---
# 使用 CacheDataset 加速训练 (将数据缓存在内存中)
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

# --- 5. 模型、损失函数、优化器 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=num_classes,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm="batch",
).to(device)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# --- 6. 训练循环 (Training Loop) ---
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

print("Start training...")
for epoch in range(max_epochs):
    model.train()
    epoch_loss = 0
    step = 0
    
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"Epoch {epoch + 1}/{max_epochs}, average loss: {epoch_loss:.4f}")

    # --- 验证循环 (Validation Loop) ---
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                
                # 使用滑动窗口推断 (Sliding Window Inference) 处理大尺寸验证图
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, 4, model
                )
                
                # 转换输出为离散标签以便计算 Dice
                val_outputs = [AsDiscrete(argmax=True, to_onehot=num_classes)(i) for i in decollate_batch(val_outputs)]
                val_labels = [AsDiscrete(to_onehot=num_classes)(i) for i in decollate_batch(val_labels)]
                
                dice_metric(y_pred=val_outputs, y=val_labels)
            
            # 聚合所有 Batch 的结果
            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            metric_values.append(metric)
            
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(
                    model.state_dict(), 
                    os.path.join(log_dir, "best_metric_model.pth")
                )
                print("Saved new best metric model!")
            
            print(
                f"current epoch: {epoch + 1} "
                f"current mean dice: {metric:.4f} "
                f"best mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )

print(f"Train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

# 绘制 Loss 曲线并保存
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
plt.xlabel("epoch")
plt.plot(epoch_loss_values)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
plt.xlabel("epoch")
plt.plot(range(val_interval, max_epochs + 1, val_interval), metric_values)
plt.savefig(os.path.join(log_dir, "loss_curve.png"))
