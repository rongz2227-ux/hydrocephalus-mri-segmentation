import os
import torch
import glob
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, 
    ScaleIntensityd, EnsureTyped, SaveImaged
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference

# --- 1. 配置路径 ---
data_dir = "./data"  
output_dir = "./ensemble_predictions" 
# 请确保这两个路径是正确的
model_swin_path = "./experiments_swin/best_metric_model_swin.pth"
model_large_path = "./experiments_large/best_metric_model_large.pth"

# 自动检测设备 (如果没有GPU，就用CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 6

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def main():
    print(f"Running inference on device: {device}")
    
    # --- 2. 准备验证数据 (取最后9个) ---
    subjects = sorted(glob.glob(os.path.join(data_dir, "sub_*")))
    val_files = []
    # 这里假设最后9个是验证集，和之前保持一致
    for sub in subjects[-9:]: 
        # 查找 T1 图像
        img_paths = glob.glob(os.path.join(sub, "scan_*", "*eT1W_FFE_SVR.nii.gz"))
        if len(img_paths) > 0:
            val_files.append({"image": img_paths[0], "name": os.path.basename(sub)})
        else:
            print(f"Warning: No image found for {sub}")

    print(f"Found {len(val_files)} subjects for inference.")

    val_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
        ScaleIntensityd(keys=["image"]),
        EnsureTyped(keys=["image"]),
    ])
    
    val_ds = Dataset(data=val_files, transform=val_transforms)
    # CPU 跑的时候 batch_size=1 最稳
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # --- 3. 创建模型 (先创建！) ---
    print("Creating models...")
    model_swin = SwinUNETR(img_size=(96, 96, 96), in_channels=1, out_channels=num_classes, feature_size=48).to(device)
    model_large = SwinUNETR(img_size=(128, 128, 128), in_channels=1, out_channels=num_classes, feature_size=48).to(device)
    
    # --- 4. 加载权重 (后加载！) ---
    print(f"Loading weights from {model_swin_path}...")
    model_swin.load_state_dict(torch.load(model_swin_path, map_location=device))
    
    print(f"Loading weights from {model_large_path}...")
    model_large.load_state_dict(torch.load(model_large_path, map_location=device))
    
    model_swin.eval()
    model_large.eval()

    saver = SaveImaged(keys="pred", meta_keys="image_meta_dict", output_dir=output_dir, output_postfix="seg", resample=False)

    # --- 5. 推理并保存 ---
    print("Starting inference...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inputs = batch["image"].to(device)
            subject_name = batch["name"][0]
            print(f"Processing {subject_name} ({i+1}/{len(val_loader)})...")
            
            # Swin 推理
            out1 = sliding_window_inference(inputs, (96, 96, 96), 4, model_swin, overlap=0.5)
            prob1 = torch.softmax(out1, dim=1)
            
            # Large 推理
            out2 = sliding_window_inference(inputs, (128, 128, 128), 1, model_large, overlap=0.5)
            prob2 = torch.softmax(out2, dim=1)
            
            # 集成 (取平均)
            avg_prob = (prob1 + prob2) / 2.0
            pred = torch.argmax(avg_prob, dim=1, keepdim=True)
            
            batch["pred"] = pred
            
            # 保存
            for item in decollate_batch(batch):
                saver(item)
                
    print(f"All done! Results saved to {output_dir}")

if __name__ == "__main__":
    main()