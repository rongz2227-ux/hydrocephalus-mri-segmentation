import os
import torch
import glob
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, 
    ScaleIntensityd, EnsureTyped, AsDiscrete
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

# ================= 配置区 =================
data_dir = "/home/lizhaolab/projects/Hydro_Seg_Project/data"

# 模型 A: Swin Baseline (96x96x96)
path_swin = "/home/lizhaolab/projects/Hydro_Seg_Project/experiments_swin/best_metric_model_swin.pth"

# 模型 B: Large ROI Swin (128x128x128)
path_large = "/home/lizhaolab/projects/Hydro_Seg_Project/experiments_large/best_metric_model_large.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 6
# =========================================

def main():
    print(f"--- Starting FINAL ENSEMBLE (Swin Baseline + Large ROI) ---")
    
    # 1. 准备数据
    subjects = sorted(glob.glob(os.path.join(data_dir, "sub_*")))
    data_dicts = []
    for sub in subjects:
        scan_dirs = sorted(glob.glob(os.path.join(sub, "scan_*")))
        if not scan_dirs: continue
        img_path = os.path.join(scan_dirs[0], "eT1W_FFE_SVR.nii.gz")
        lbl_path = os.path.join(scan_dirs[0], "segmentation", "gt.nii.gz")
        if os.path.exists(img_path) and os.path.exists(lbl_path):
            data_dicts.append({"image": img_path, "label": lbl_path})
    
    val_files = data_dicts[-9:]
    print(f"Validation samples: {len(val_files)}")

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

    # 2. 加载模型 A
    print(f"Loading Swin Baseline (ROI 96)...")
    model_a = SwinUNETR(
        img_size=(96, 96, 96), in_channels=1, out_channels=num_classes,
        feature_size=48, use_checkpoint=False
    ).to(device)
    
    if os.path.exists(path_swin):
        state_a = torch.load(path_swin, map_location=device)
        new_state_a = {k.replace("module.", ""): v for k, v in state_a.items()}
        model_a.load_state_dict(new_state_a)
        model_a.eval()
    else:
        print(f"ERROR: Model A not found at {path_swin}")
        return

    # 3. 加载模型 B
    print(f"Loading Large ROI Swin (ROI 128)...")
    model_b = SwinUNETR(
        img_size=(128, 128, 128), in_channels=1, out_channels=num_classes,
        feature_size=48, use_checkpoint=False
    ).to(device)
    
    if os.path.exists(path_large):
        state_b = torch.load(path_large, map_location=device)
        new_state_b = {k.replace("module.", ""): v for k, v in state_b.items()}
        model_b.load_state_dict(new_state_b)
        model_b.eval()
    else:
        print(f"ERROR: Model B not found at {path_large}")
        return

    # 4. 集成推理
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    
    print("Running Ensemble Inference (Low Memory Mode)...")
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data["image"].to(device), data["label"].to(device)
            
            # --- 预测 A (Baseline) ---
            # 【关键修改】sw_batch_size 改为 2 (96尺寸较小，2没问题)
            out_a = sliding_window_inference(inputs, (96, 96, 96), 2, model_a, overlap=0.5)
            prob_a = torch.softmax(out_a, dim=1)
            
            # --- 预测 B (Large ROI) ---
            # 【关键修改】sw_batch_size 改为 1 (128尺寸巨大，必须为1)
            out_b = sliding_window_inference(inputs, (128, 128, 128), 1, model_b, overlap=0.5)
            prob_b = torch.softmax(out_b, dim=1)
            
            # --- 加权融合 ---
            avg_prob = (0.5 * prob_a) + (0.5 * prob_b)
            
            val_outputs = [AsDiscrete(argmax=True, to_onehot=num_classes)(j) for j in decollate_batch(avg_prob)]
            val_labels = [AsDiscrete(to_onehot=num_classes)(j) for j in decollate_batch(labels)]
            
            dice_metric(y_pred=val_outputs, y=val_labels)
            print(f"Processed sample {i+1}/{len(val_files)}...", end="\r")

    # 5. 结果报告
    metric_batch = dice_metric.aggregate()
    final_mean = torch.mean(metric_batch).item()
    
    class_names = ["CSF/SAS", "GrayMatter", "WhiteMatter", "Ventricle", "ChoroidPlexus"]
    
    print("\n" + "="*40)
    print(f"FINAL ENSEMBLE RESULTS (Swin + Large)")
    print("="*40)
    print(f"Overall Mean Dice: {final_mean:.4f}")
    print("-" * 20)
    for i, name in enumerate(class_names):
        print(f"Index {i+1} {name}: {metric_batch[i].item():.4f}")
    print("="*40)

if __name__ == "__main__":
    main()