import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    AddChanneld,        # 用于 Image/Label
    AsChannelFirstd,    # 【新增】用于处理 Prob Map 的通道顺序
    ConcatItemsd,        
    ResizeWithPadOrCropd
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.data import (
    DataLoader,
    CacheDataset,
    decollate_batch,
)

# ================= 配置部分 =================
data_dir = "/home/lizhaolab/projects/Hydro_Seg_Project/data"
prob_map_path = os.path.join(data_dir, "probability_map.nii.gz")
log_dir = "/home/lizhaolab/projects/Hydro_Seg_Project/experiments_prob"

# 路径检查
print(f"DEBUG: Data Directory: {data_dir}")
if not os.path.exists(prob_map_path):
    raise FileNotFoundError(f"CRITICAL: Probability map not found at {prob_map_path}")
else:
    print(f"DEBUG: Probability map found.")

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

roi_size = (96, 96, 96)
batch_size = 2          
sw_batch_size = 4
max_epochs = 300
val_every = 5
num_classes = 6       
# ===========================================

# -------------------------------------------------------------------
# Transforms (维度逻辑已修复)
# -------------------------------------------------------------------
train_transforms = Compose(
    [
        # 1. 加载图像
        LoadImaged(keys=["image", "label", "prob"]),
        
        # 2. 通道处理 (分情况处理)
        # Image/Label 是 3D (H,W,D) -> 加一个通道变 (1,H,W,D)
        AddChanneld(keys=["image", "label"]),
        
        # Prob 是 4D (H,W,D,2) -> 把最后的通道移到最前变 (2,H,W,D)
        AsChannelFirstd(keys=["prob"], channel_dim=-1),
        
        # 3. 统一方向 (此时所有数据都是 C,H,W,D 格式，Orientationd 能正确识别)
        Orientationd(keys=["image", "label", "prob"], axcodes="RAS"),
        
        # 4. 统一分辨率
        Spacingd(
            keys=["image", "label", "prob"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest", "bilinear"),
        ),
        
        # 5. 强度归一化
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=0.0, a_max=255.0, 
            b_min=0.0, b_max=1.0, 
            clip=True,
        ),
        
        # 6. 对齐大小
        ResizeWithPadOrCropd(keys=["image", "label", "prob"], spatial_size=(256, 256, 256)),

        # 7. 拼接通道: Image(1) + Prob(2) = NewImage(3)
        ConcatItemsd(keys=["image", "prob"], name="image", dim=0),

        # 8. 随机裁剪
        RandCropByPosNegLabeld(
            keys=["image", "label"], 
            label_key="label",
            spatial_size=roi_size,
            pos=2,
            neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
        ),
        
        # 9. 数据增强
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
        
        EnsureTyped(keys=["image", "label"]),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label", "prob"]),
        
        # 验证集同样的通道处理逻辑
        AddChanneld(keys=["image", "label"]),
        AsChannelFirstd(keys=["prob"], channel_dim=-1),
        
        Orientationd(keys=["image", "label", "prob"], axcodes="RAS"),
        Spacingd(keys=["image", "label", "prob"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest", "bilinear")),
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        ResizeWithPadOrCropd(keys=["image", "label", "prob"], spatial_size=(256, 256, 256)),
        ConcatItemsd(keys=["image", "prob"], name="image", dim=0),
        EnsureTyped(keys=["image", "label"]),
    ]
)

# -------------------------------------------------------------------
# 主程序
# -------------------------------------------------------------------
def main():
    # --- 数据加载 ---
    print(f"Searching for subjects in {data_dir}...")
    subjects = sorted(glob.glob(os.path.join(data_dir, "sub_*")))
    
    data_dicts = []
    for sub in subjects:
        scan_dirs = sorted(glob.glob(os.path.join(sub, "scan_*")))
        if not scan_dirs: continue
        
        scan_path = scan_dirs[0] 
        img_path = os.path.join(scan_path, "eT1W_FFE_SVR.nii.gz")
        lbl_path = os.path.join(scan_path, "segmentation", "gt.nii.gz")
        
        if os.path.exists(img_path) and os.path.exists(lbl_path):
            data_dicts.append({
                "image": img_path, 
                "label": lbl_path,
                "prob": prob_map_path
            })
    
    train_files, val_files = data_dicts[:-9], data_dicts[-9:]
    print(f"DEBUG: Total valid subjects found: {len(data_dicts)}")
    print(f"Train samples: {len(train_files)} | Val samples: {len(val_files)}")
    
    if len(train_files) == 0:
        raise ValueError("No training files found!")

    # --- Dataset & DataLoader ---
    train_ds = CacheDataset(
        data=train_files, transform=train_transforms,
        cache_rate=1.0, num_workers=2
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, 
        cache_rate=1.0, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2
    )

    # --- 模型定义 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SwinUNETR(
        img_size=roi_size,
        in_channels=3,     # 1 Image + 2 Prob
        out_channels=num_classes, 
        feature_size=48,
        use_checkpoint=True,
    ).to(device)

    # 加载权重
    weight_path = "/home/lizhaolab/projects/Hydro_Seg_Project/model_swinvit.pt"
    if os.path.exists(weight_path):
        print("Loading pre-trained weights...")
        try:
            checkpoint = torch.load(weight_path)
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            
            model_dict = model.state_dict()
            full_dict = {}
            for k, v in state_dict.items():
                target_key = "swinViT." + k if not k.startswith("module.") else k.replace("module.", "swinViT.", 1)
                
                if "patch_embed" in target_key: 
                    continue
                    
                if target_key in model_dict and v.shape == model_dict[target_key].shape:
                    full_dict[target_key] = v
            
            model.load_state_dict(model_dict, strict=False)
            print(f"Loaded {len(full_dict)} layers from pre-trained weights.")
        except Exception as e:
            print(f"Error loading weights: {e}")

    # --- 训练参数 ---
    class_weights = torch.tensor([0.1, 2.0, 1.0, 1.0, 3.0, 5.0]).to(device)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, ce_weight=class_weights, lambda_dice=1.0, lambda_ce=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")

    # --- 训练循环 ---
    best_metric = -1
    best_metric_epoch = -1
    
    print("Start Training...")
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

        if (epoch + 1) % val_every == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model, overlap=0.5)
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
                    torch.save(model.state_dict(), os.path.join(log_dir, "best_metric_model_prob.pth"))
                    print("   [SAVED] New Best Model!")
                
                dice_metric_batch.reset()

if __name__ == "__main__":
    main()