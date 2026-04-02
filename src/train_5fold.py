import os
import glob
import torch
import numpy as np
import argparse
from sklearn.model_selection import KFold
from monai.utils import set_determinism
from monai.transforms import (
    Compose, LoadImaged, Orientationd, Spacingd, 
    ScaleIntensityd, RandCropByPosNegLabeld, RandFlipd, RandRotate90d, 
    Rand3DElasticd, EnsureTyped, AsDiscrete, AddChanneld
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

# --- 命令行参数 ---
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=0, help="Fold index (0-4)")
args = parser.parse_args()

fold_idx = args.fold
print(f"=== FOLD {fold_idx} STARTING (Low Memory Mode) ===")

# --- 配置 ---
data_dir = "/home/lizhaolab/projects/Hydro_Seg_Project/data"
log_dir = f"/home/lizhaolab/projects/Hydro_Seg_Project/experiments_5fold/fold_{fold_idx}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

max_epochs = 250
val_interval = 5

# 【关键修改 1】真实 Batch Size = 1 * 1 = 1
batch_size = 1
# 【关键修改 2】因为每次只看1个块，我们要把“积攒”步数增加到 4，模拟大 Batch 的效果
gradient_accumulation_steps = 4 

lr = 1e-4
roi_size = (96, 96, 96) 
num_classes = 6

set_determinism(seed=42)

# --- 1. 数据准备 ---
subjects = sorted(glob.glob(os.path.join(data_dir, "sub_*")))
all_files = []
for sub in subjects:
    scan_dirs = sorted(glob.glob(os.path.join(sub, "scan_*")))
    if not scan_dirs: continue
    img_path = os.path.join(scan_dirs[0], "eT1W_FFE_SVR.nii.gz")
    lbl_path = os.path.join(scan_dirs[0], "segmentation", "gt.nii.gz")
    if os.path.exists(img_path) and os.path.exists(lbl_path):
        all_files.append({"image": img_path, "label": lbl_path})

# K-Fold 划分
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_files = []
val_files = []
for i, (train_idx, val_idx) in enumerate(kf.split(all_files)):
    if i == fold_idx:
        train_files = [all_files[j] for j in train_idx]
        val_files = [all_files[j] for j in val_idx]
        break

# --- 2. Transforms ---
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    AddChanneld(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityd(keys=["image"]),
    # 【关键修改 3】num_samples 必须改为 1！否则 GPU 还是会爆！
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label", spatial_size=roi_size,
        pos=2, neg=1, num_samples=1, image_key="image", image_threshold=0,
    ),
    Rand3DElasticd(keys=["image", "label"], sigma_range=(5, 8), magnitude_range=(100, 200), prob=0.2, mode=("bilinear", "nearest")),
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.1),
    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.1),
    RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.1),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    AddChanneld(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityd(keys=["image"]),
    EnsureTyped(keys=["image", "label"]),
])

# DataLoader
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=2)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

# --- 3. 模型 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinUNETR(
    img_size=roi_size, in_channels=1, out_channels=num_classes,
    feature_size=48, use_checkpoint=True,
).to(device)

# 加载预训练权重 (尝试)
weight_path = "/home/lizhaolab/projects/Hydro_Seg_Project/model_swinvit.pt"
if os.path.exists(weight_path):
    try:
        checkpoint = torch.load(weight_path)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model_dict = model.state_dict()
        full_dict = {}
        for k, v in state_dict.items():
            target_key = "swinViT." + k if not k.startswith("module.") else k.replace("module.", "swinViT.", 1)
            if target_key in model_dict and v.shape == model_dict[target_key].shape:
                full_dict[target_key] = v
        model.load_state_dict(full_dict, strict=False)
        print("Loaded weights.")
    except: pass

# --- 4. Loss & Optimizer ---
class_weights = torch.tensor([0.1, 10.0, 0.1, 0.1, 10.0, 5.0]).to(device)
loss_function = DiceCELoss(to_onehot_y=True, softmax=True, ce_weight=class_weights, lambda_dice=1.0, lambda_ce=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")

# --- 5. 训练循环 ---
best_metric = -1
print("Start Training...")

for epoch in range(max_epochs):
    model.train()
    epoch_loss = 0
    step = 0
    optimizer.zero_grad() 
    
    for i, batch_data in enumerate(train_loader):
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        
        # 梯度累积
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        epoch_loss += loss.item() * gradient_accumulation_steps

    epoch_loss /= step
    
    # 验证
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
            # 0=CSF, 3=Ventricle (因为不含背景)
            csf_score = metric_batch[0].item()
            vent_score = metric_batch[3].item()
            target_metric = (csf_score + vent_score) / 2.0
            
            dice_metric_batch.reset()
            
            print(f"Fold {fold_idx} | Ep {epoch + 1} | Target: {target_metric:.4f} (CSF: {csf_score:.4f}, Vent: {vent_score:.4f})")
            
            if target_metric > best_metric:
                best_metric = target_metric
                # 覆盖保存 best_model，节省空间
                torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
                print(f"   [SAVED]")

print(f"Fold {fold_idx} Done.")