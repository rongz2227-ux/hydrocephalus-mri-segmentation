import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    Rand3DElasticd,
    EnsureTyped,
    AsDiscrete,
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

# --- 1. 配置参数 ---
data_dir = os.path.expanduser("~/projects/Hydro_Seg_Project/data")
log_dir = os.path.expanduser("~/projects/Hydro_Seg_Project/experiments_opt")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 训练参数
max_epochs = 200
val_interval = 5
batch_size = 2
lr = 1e-4
roi_size = (96, 96, 96)
num_classes = 6

set_determinism(seed=42)

# --- 2. 数据准备 ---
print(f"Searching data in: {data_dir}")
subjects = sorted(glob.glob(os.path.join(data_dir, "sub_*")))
data_dicts = []
for sub in subjects:
    scan_dirs = sorted(glob.glob(os.path.join(sub, "scan_*")))
    if len(scan_dirs) == 0: continue
    scan_path = scan_dirs[0]
    img_path = os.path.join(scan_path, "eT1W_FFE_SVR.nii.gz")
    lbl_path = os.path.join(scan_path, "segmentation", "gt.nii.gz")
    if os.path.exists(img_path) and os.path.exists(lbl_path):
        data_dicts.append({"image": img_path, "label": lbl_path})

train_files, val_files = data_dicts[:-9], data_dicts[-9:]
print(f"Train: {len(train_files)} | Val: {len(val_files)}")

# --- 3. 数据增强策略 ---
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityd(keys=["image"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            pos=2, neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
        ),
        # 弹性变形 (针对脑积水优化)
        Rand3DElasticd(
            keys=["image", "label"],
            sigma_range=(5, 8),
            magnitude_range=(100, 200),
            prob=0.2,
            mode=("bilinear", "nearest"),
        ),
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
# num_workers=2 比较稳妥，防止内存溢出
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=2)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

# --- 5. 模型与损失函数 ---
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

# 权重定义：背景0.1, CSF 2.0, 灰质1.0, 白质1.0, 脑室3.0, 脉络丛5.0
class_weights = torch.tensor([0.1, 2.0, 1.0, 1.0, 3.0, 5.0]).to(device)

loss_function = DiceCELoss(
    to_onehot_y=True, 
    softmax=True,
    # 【注意】MONAI 0.9.1 使用 ce_weight 参数
    ce_weight=class_weights,  
    lambda_dice=1.0,
    lambda_ce=1.0
)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch") 

# --- 6. 训练循环 ---
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

print("Start OPTIMIZED training...")
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
    print(f"Epoch {epoch + 1}/{max_epochs}, loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                
                # overlap=0.5 消除棋盘格效应
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, 4, model, overlap=0.5
                )
                
                val_outputs = [AsDiscrete(argmax=True, to_onehot=num_classes)(i) for i in decollate_batch(val_outputs)]
                val_labels = [AsDiscrete(to_onehot=num_classes)(i) for i in decollate_batch(val_labels)]
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

            metric_batch = dice_metric_batch.aggregate()
            metric = torch.mean(metric_batch).item() 
            metric_values.append(metric)
            
            class_names = ["CSF", "GrayMatter", "WhiteMatter", "Ventricle", "ChoroidPlexus"]
            print(f"\n>> Epoch {epoch + 1} Report:")
            print(f"   Mean Dice: {metric:.4f}")
            for i, name in enumerate(class_names):
                print(f"   - {name}: {metric_batch[i].item():.4f}")

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(log_dir, "best_metric_model_opt.pth"))
                print("   [SAVED] New Best Model!")
            
            dice_metric_batch.reset()

print(f"Optimization done. Best Dice: {best_metric:.4f} at epoch {best_metric_epoch}")