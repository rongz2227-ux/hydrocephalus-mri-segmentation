import os
import torch
import matplotlib.pyplot as plt
import numpy as np
# 【关键修改】引入 SwinUNETR
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityd, EnsureTyped
)
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference

# --- 配置 ---
data_dir = os.path.expanduser("~/projects/Hydro_Seg_Project/data")
# 确保这里指向 Swin 的最佳模型
model_path = os.path.expanduser("~/projects/Hydro_Seg_Project/experiments_swin/best_metric_model_swin.pth")

# 自动检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running visualization on: {device}")

# --- 1. 加载一个验证样本 ---
# 随便找一个样本，比如 sub_045
test_files = [{"image": f"{data_dir}/sub_045/scan_01/eT1W_FFE_SVR.nii.gz", 
               "label": f"{data_dir}/sub_045/scan_01/segmentation/gt.nii.gz"}]

# 如果文件不存在，切回 sub_001
if not os.path.exists(test_files[0]["image"]):
    print(f"Test file not found, switching to sub_001...")
    test_files = [{"image": f"{data_dir}/sub_001/scan_01/eT1W_FFE_SVR.nii.gz", 
                   "label": f"{data_dir}/sub_001/scan_01/segmentation/gt.nii.gz"}]

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityd(keys=["image"]),
    EnsureTyped(keys=["image", "label"]),
])

test_ds = Dataset(data=test_files, transform=val_transforms)
test_loader = DataLoader(test_ds, batch_size=1)

# --- 2. 加载模型 (必须与训练时完全一致) ---
print(f"Loading model architecture: SwinUNETR...")
# 【关键修改】这里必须是 SwinUNETR，且 feature_size=48
model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=6,
    feature_size=48, 
    use_checkpoint=False # 推理时不需要 checkpoint
).to(device)

print(f"Loading weights from {model_path}...")
try:
    # 加载权重 (自动适配 CPU/GPU)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Weights loaded successfully!")
except Exception as e:
    print(f"Error loading weights: {e}")
    exit(1)

model.eval()

# --- 3. 推理与绘图 ---
print("Starting inference...")
with torch.no_grad():
    for data in test_loader:
        inputs = data["image"].to(device)
        labels = data["label"].to(device)
        
        # 推理 (使用重叠滑动窗口消除拼缝)
        print("Running sliding window inference (this may take a moment)...")
        outputs = sliding_window_inference(inputs, (96, 96, 96), 4, model, overlap=0.5)
        
        # 取 argmax 得到分类结果
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        
        img_np = inputs.cpu().numpy()
        lbl_np = labels.cpu().numpy() 
        
        # 绘图：取中间切片
        slice_idx = img_np.shape[4] // 2
        print(f"Plotting slice {slice_idx}...")
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("MRI Image")
        plt.imshow(img_np[0, 0, :, :, slice_idx], cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(lbl_np[0, 0, :, :, slice_idx], cmap="jet", vmin=0, vmax=5)
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title(f"Swin Prediction (Dice ~0.72)")
        plt.imshow(preds[0, :, :, slice_idx], cmap="jet", vmin=0, vmax=5)
        plt.axis("off")
        
        save_name = "swin_result_preview.png"
        plt.savefig(save_name)
        print(f"Preview saved to {os.path.abspath(save_name)}")
        print("Done!")
        break