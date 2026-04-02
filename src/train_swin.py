import os
import glob
import torch
import numpy as np
from monai.utils import set_determinism
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, 
    ScaleIntensityd, RandCropByPosNegLabeld, RandFlipd, RandRotate90d, 
    Rand3DElasticd, EnsureTyped, AsDiscrete
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

# --- 1. 配置参数 ---
data_dir = os.path.expanduser("~/projects/Hydro_Seg_Project/data")
log_dir = os.path.expanduser("~/projects/Hydro_Seg_Project/experiments_swin")
if not os.path.exists(log_dir): os.makedirs(log_dir)

# 训练参数
max_epochs = 300       # 300轮足以让 Transformer 收敛
val_interval = 5
batch_size = 2
lr = 1e-4
roi_size = (96, 96, 96)
num_classes = 6

set_determinism(seed=42)

# --- 2. 数据准备 (带文件检查) ---
print(f"Searching data in: {data_dir}")
subjects = sorted(glob.glob(os.path.join(data_dir, "sub_*")))
data_dicts = []
for sub in subjects:
    scan_dirs = sorted(glob.glob(os.path.join(sub, "scan_*")))
    if len(scan_dirs) == 0: continue
    
    scan_path = scan_dirs[0]
    img_path = os.path.join(scan_path, "eT1W_FFE_SVR.nii.gz")
    lbl_path = os.path.join(scan_path, "segmentation", "gt.nii.gz")
    
    # 只有文件都存在才加入训练列表
    if os.path.exists(img_path) and os.path.exists(lbl_path):
        data_dicts.append({"image": img_path, "label": lbl_path})
    else:
        print(f"Skipping {sub}: missing files.")

train_files, val_files = data_dicts[:-9], data_dicts[-9:]
print(f"Total Valid Subjects: {len(data_dicts)}")
print(f"Train: {len(train_files)} | Val: {len(val_files)}")

# --- 3. 数据增强策略 (高强度) ---
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityd(keys=["image"]),
    # 随机裁剪
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label", spatial_size=roi_size,
        pos=2, neg=1, num_samples=2, image_key="image", image_threshold=0,
    ),
    # 弹性变形 (模拟脑积水形状)
    Rand3DElasticd(keys=["image", "label"], sigma_range=(5, 8), magnitude_range=(100, 200), prob=0.2, mode=("bilinear", "nearest")),
    # 翻转与旋转
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.1),
    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.1),
    RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.1),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityd(keys=["image"]),
    EnsureTyped(keys=["image", "label"]),
])

# DataLoader (num_workers=2 防止死锁)
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=2)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

# --- 4. 模型定义与预训练权重加载 (核心修复) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | Model: SwinUNETR (SOTA)")

# 【修复1】feature_size=48 以匹配官方权重
model = SwinUNETR(
    img_size=roi_size,
    in_channels=1,
    out_channels=num_classes,
    feature_size=48, 
    use_checkpoint=True, 
).to(device)

# 加载预训练权重
weight_path = os.path.expanduser("~/projects/Hydro_Seg_Project/model_swinvit.pt")
if os.path.exists(weight_path):
    print(f"Loading pre-trained weights from {weight_path}...")
    try:
        checkpoint = torch.load(weight_path)
        if "state_dict" in checkpoint:
            pretrained_dict = checkpoint["state_dict"]
        else:
            pretrained_dict = checkpoint

        model_dict = model.state_dict()
        full_dict = {}
        
        # 【修复2】智能 Key 映射逻辑
        for k, v in pretrained_dict.items():
            # 官方权重通常叫 "module.patch_embed..."
            # MONAI 模型需要 "swinViT.patch_embed..."
            
            # 1. 去掉 "module." 前缀
            if k.startswith("module."):
                base_key = k[7:]
            else:
                base_key = k
            
            # 2. 加上 "swinViT." 前缀
            target_key = "swinViT." + base_key

            # 3. 尝试匹配
            if target_key in model_dict:
                if v.shape == model_dict[target_key].shape:
                    full_dict[target_key] = v
                else:
                    print(f"Skipping {target_key}: shape mismatch {v.shape} vs {model_dict[target_key].shape}")
            else:
                # 某些不需要的前缀（如 norm, head）忽略即可
                pass
        
        if len(full_dict) > 0:
            model_dict.update(full_dict)
            model.load_state_dict(model_dict)
            print(f"SUCCESS: Loaded {len(full_dict)} layers from pre-trained weights!")
        else:
            print("WARNING: Loaded 0 layers! Check key mapping.")
            
    except Exception as e:
        print(f"ERROR loading weights: {e}")
else:
    print("Warning: Pre-trained weight file not found. Training from scratch.")

# --- 5. Loss, Optimizer, Metric ---
# 权重: 背景0.1, CSF 2.0, 灰质1.0, 白质1.0, 脑室3.0, 脉络丛5.0
class_weights = torch.tensor([0.1, 2.0, 1.0, 1.0, 3.0, 5.0]).to(device)

loss_function = DiceCELoss(
    to_onehot_y=True, 
    softmax=True, 
    ce_weight=class_weights,  # MONAI 0.9.x 使用 ce_weight
    lambda_dice=1.0, 
    lambda_ce=1.0
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")

# --- 6. 训练循环 ---
best_metric = -1
best_metric_epoch = -1
print("Start Swin-UNETR training...")

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
    print(f"Epoch {epoch + 1}/{max_epochs}, loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                val_outputs = sliding_window_inference(val_inputs, roi_size, 4, model, overlap=0.5)
                val_outputs = [AsDiscrete(argmax=True, to_onehot=num_classes)(i) for i in decollate_batch(val_outputs)]
                val_labels = [AsDiscrete(to_onehot=num_classes)(i) for i in decollate_batch(val_labels)]
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

            metric_batch = dice_metric_batch.aggregate()
            metric = torch.mean(metric_batch).item()
            
            class_names = ["CSF", "GrayMatter", "WhiteMatter", "Ventricle", "ChoroidPlexus"]
            print(f">> Epoch {epoch + 1} | Mean Dice: {metric:.4f}")
            for i, name in enumerate(class_names):
                print(f"   - {name}: {metric_batch[i].item():.4f}")

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(log_dir, "best_metric_model_swin.pth"))
                print("   [SAVED] New Best Swin Model!")
            dice_metric_batch.reset()

print(f"Training Done. Best Dice: {best_metric:.4f}")