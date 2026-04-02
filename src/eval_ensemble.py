import os
import torch
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, 
    ScaleIntensityd, EnsureTyped, AsDiscrete
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import UNet, SwinUNETR
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

# --- 配置 ---
data_dir = os.path.expanduser("~/projects/Hydro_Seg_Project/data")
# U-Net 权重路径
unet_path = os.path.expanduser("~/projects/Hydro_Seg_Project/experiments_opt/best_metric_model_opt.pth")
# Swin 权重路径
swin_path = os.path.expanduser("~/projects/Hydro_Seg_Project/experiments_swin/best_metric_model_swin.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
roi_size = (96, 96, 96)
num_classes = 6

# --- 1. 数据 ---
import glob
subjects = sorted(glob.glob(os.path.join(data_dir, "sub_*")))
data_dicts = []
for sub in subjects:
    scan_dirs = sorted(glob.glob(os.path.join(sub, "scan_*")))
    if len(scan_dirs) == 0: continue
    data_dicts.append({
        "image": os.path.join(scan_dirs[0], "eT1W_FFE_SVR.nii.gz"),
        "label": os.path.join(scan_dirs[0], "segmentation", "gt.nii.gz")
    })
val_files = data_dicts[-9:] # 验证集

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityd(keys=["image"]),
    EnsureTyped(keys=["image", "label"]),
])
val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

# --- 2. 加载两个模型 ---
print("Loading Models...")

# Model 1: U-Net
model_unet = UNet(
    spatial_dims=3, in_channels=1, out_channels=num_classes,
    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2, norm="batch",
).to(device)
model_unet.load_state_dict(torch.load(unet_path))
model_unet.eval()

# Model 2: Swin UNETR
model_swin = SwinUNETR(
    img_size=roi_size, in_channels=1, out_channels=num_classes,
    feature_size=48, use_checkpoint=False,
).to(device)
model_swin.load_state_dict(torch.load(swin_path))
model_swin.eval()

# --- 3. 集成推理 ---
dice_metric = DiceMetric(include_background=False, reduction="mean")

print("Starting Ensemble Evaluation...")
with torch.no_grad():
    for data in val_loader:
        inputs, labels = data["image"].to(device), data["label"].to(device)
        
        # 1. U-Net 预测 (Softmax 概率)
        out_unet = sliding_window_inference(inputs, roi_size, 4, model_unet, overlap=0.5)
        prob_unet = torch.softmax(out_unet, dim=1)
        
        # 2. Swin 预测 (Softmax 概率)
        out_swin = sliding_window_inference(inputs, roi_size, 4, model_swin, overlap=0.5)
        prob_swin = torch.softmax(out_swin, dim=1)
        
        # 3. 【核心步骤】取平均 (Ensemble)
        # 可以给 Swin 更高的权重，比如 0.6 * Swin + 0.4 * Unet
        avg_prob = (0.4 * prob_unet) + (0.6 * prob_swin)
        
        # 4. 离散化并计算分数
        val_outputs = [AsDiscrete(argmax=True, to_onehot=num_classes)(i) for i in decollate_batch(avg_prob)]
        val_labels = [AsDiscrete(to_onehot=num_classes)(i) for i in decollate_batch(labels)]
        
        dice_metric(y_pred=val_outputs, y=val_labels)

    final_metric = dice_metric.aggregate().item()
    print(f"Ensemble Mean Dice: {final_metric:.4f}")