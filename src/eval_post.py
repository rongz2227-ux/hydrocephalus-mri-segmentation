import os
import torch
import argparse
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, 
    ScaleIntensityd, EnsureTyped, AsDiscrete, KeepLargestConnectedComponent
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

# --- 配置 ---
data_dir = os.path.expanduser("~/projects/Hydro_Seg_Project/data")
# 确保指向最好的 Swin 模型
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="~/projects/Hydro_Seg_Project/experiments_swin/best_metric_model_swin.pth")
args = parser.parse_args()
model_path = os.path.expanduser(args.model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# 只取验证集
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
    img_size=(96, 96, 96), in_channels=1, out_channels=6, feature_size=48, use_checkpoint=False
).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# --- 3. 定义后处理变换 ---
# 这是一个强大的“清洁工”，只保留最大的连通块
post_process = KeepLargestConnectedComponent(applied_labels=[1, 4, 5]) 
# 只对 CSF(1), 脑室(4), 脉络丛(5) 做清理，灰白质连通性复杂，不动

dice_metric = DiceMetric(include_background=False, reduction="mean")
dice_metric_raw = DiceMetric(include_background=False, reduction="mean")
dice_v_raw = DiceMetric(include_background=False, reduction="mean")
dice_v_post = DiceMetric(include_background=False, reduction="mean")

print("Starting Evaluation with Post-processing...")
with torch.no_grad():
    for data in val_loader:
        inputs, labels = data["image"].to(device), data["label"].to(device)
        
        # 推理
        outputs = sliding_window_inference(inputs, (96, 96, 96), 4, model, overlap=0.5)
        
        # 1. 原始结果 (离散化)
        outputs_raw = [AsDiscrete(argmax=True, to_onehot=6)(i) for i in decollate_batch(outputs)]
        labels_list = [AsDiscrete(to_onehot=6)(i) for i in decollate_batch(labels)]
        
        # 计算原始分数
        dice_metric_raw(y_pred=outputs_raw, y=labels_list)
        # ---- Ventricle Dice (label=4) on RAW ----
        v_raw = [i[4:5] for i in outputs_raw]
        v_lab = [i[4:5] for i in labels_list]
        dice_v_raw(y_pred=v_raw, y=v_lab)

        # 2. 后处理结果
        # 对预测结果应用“最大连通域保留”
        outputs_post = [post_process(i) for i in outputs_raw]
        
        # 计算后处理分数
        dice_metric(y_pred=outputs_post, y=labels_list)
        # ---- Ventricle Dice (label=4) on POST ----
        v_post = [i[4:5] for i in outputs_post]
        dice_v_post(y_pred=v_post, y=v_lab)

    metric_raw = dice_metric_raw.aggregate().item()
    metric_post = dice_metric.aggregate().item()
    v_raw_score = dice_v_raw.aggregate().item()
    v_post_score = dice_v_post.aggregate().item()

    print("-" * 30)
    print(f"Original Mean Dice: {metric_raw:.4f}")
    print(f"Original Ventricle Dice (label=4): {v_raw_score:.4f}")
    print(f"Post-processed Ventricle Dice (label=4): {v_post_score:.4f}")
    print(f"Post-processed Dice: {metric_post:.4f}")
    print("-" * 30)
    print(f"Improvement: {(metric_post - metric_raw) * 100:.2f}%")