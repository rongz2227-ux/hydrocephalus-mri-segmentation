import os
import torch
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, 
    ScaleIntensityd, EnsureTyped, AsDiscrete
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

# --- 配置 ---
data_dir = os.path.expanduser("~/projects/Hydro_Seg_Project/data")
# 只用最强的 Swin 模型
model_path = os.path.expanduser("~/projects/Hydro_Seg_Project/experiments_swin/best_metric_model_swin.pth")
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
val_files = data_dicts[-9:]

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

# --- 2. 模型 ---
model = SwinUNETR(
    img_size=roi_size, in_channels=1, out_channels=num_classes,
    feature_size=48, use_checkpoint=False,
).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# --- 3. 定义 TTA 推理函数 ---
def tta_inference(inputs, model):
    # 1. 原始预测
    pred_raw = sliding_window_inference(inputs, roi_size, 4, model, overlap=0.5)
    pred_raw = torch.softmax(pred_raw, dim=1)
    
    # 2. 翻转预测 (水平翻转 flip dim=2)
    inputs_flip1 = torch.flip(inputs, dims=(2,))
    pred_flip1 = sliding_window_inference(inputs_flip1, roi_size, 4, model, overlap=0.5)
    pred_flip1 = torch.flip(torch.softmax(pred_flip1, dim=1), dims=(2,))
    
    # 3. 翻转预测 (前后翻转 flip dim=3)
    inputs_flip2 = torch.flip(inputs, dims=(3,))
    pred_flip2 = sliding_window_inference(inputs_flip2, roi_size, 4, model, overlap=0.5)
    pred_flip2 = torch.flip(torch.softmax(pred_flip2, dim=1), dims=(3,))
    
    # 4. 翻转预测 (上下翻转 flip dim=4)
    inputs_flip3 = torch.flip(inputs, dims=(4,))
    pred_flip3 = sliding_window_inference(inputs_flip3, roi_size, 4, model, overlap=0.5)
    pred_flip3 = torch.flip(torch.softmax(pred_flip3, dim=1), dims=(4,))

    # 平均所有结果
    avg_pred = (pred_raw + pred_flip1 + pred_flip2 + pred_flip3) / 4.0
    return avg_pred

dice_metric = DiceMetric(include_background=False, reduction="mean")

print("Starting TTA Evaluation (Self-Ensemble)...")
with torch.no_grad():
    for data in val_loader:
        inputs, labels = data["image"].to(device), data["label"].to(device)
        
        # 使用 TTA
        val_outputs = tta_inference(inputs, model)
        
        val_outputs = [AsDiscrete(argmax=True, to_onehot=num_classes)(i) for i in decollate_batch(val_outputs)]
        val_labels = [AsDiscrete(to_onehot=num_classes)(i) for i in decollate_batch(labels)]
        
        dice_metric(y_pred=val_outputs, y=val_labels)

    metric = dice_metric.aggregate().item()
    print("-" * 30)
    print(f"Original Best Dice: 0.7241")
    print(f"TTA Mean Dice:      {metric:.4f}")
    print("-" * 30)