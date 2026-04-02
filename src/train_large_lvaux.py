import os
import glob
import torch
import numpy as np
def read_subjects(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]
        
from monai.utils import set_determinism
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, 
    ScaleIntensityd, RandCropByPosNegLabeld, RandFlipd, RandRotate90d, 
    Rand3DElasticd, EnsureTyped, AsDiscrete
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

# --- 1. 配置参数 ---
data_dir = os.path.expanduser("~/projects/Hydro_Seg_Project/data")
log_dir = os.path.expanduser("~/projects/Hydro_Seg_Project/experiments_large_lvaux")
if not os.path.exists(log_dir): os.makedirs(log_dir)

# 训练策略参数
max_epochs = 600                # 给大模型充足的时间收敛
val_interval = 5
batch_size = 1                  # 【关键】单卡显存限制，只能设为1
gradient_accumulation_steps = 4 # 【关键】累积4步梯度，相当于 Batch Size = 4
lr = 1e-4
roi_size = (128, 128, 128)      # 【核心优化】扩大视野，看清全局结构
num_classes = 6

set_determinism(seed=42)

# --- 2. 数据准备 ---

print(f"Searching data in: {data_dir}")
subjects = sorted(glob.glob(os.path.join(data_dir, "sub_*")))
data_dicts = []

for sub in subjects:
    scan_dirs = sorted(glob.glob(os.path.join(sub, "scan_*")))
    if len(scan_dirs) == 0:
        continue

    scan_path = scan_dirs[0]
    img_path = os.path.join(scan_path, "eT1W_FFE_SVR.nii.gz")
    lbl_path = os.path.join(scan_path, "segmentation", "gt.nii.gz")

    if os.path.exists(img_path) and os.path.exists(lbl_path):
        data_dicts.append({"image": img_path, "label": lbl_path})
    else:
        print(f"Skipping {sub}: missing files.")

train_files, val_files = data_dicts[:-9], data_dicts[-9:]
print(f"Total: {len(data_dicts)} | Train: {len(train_files)} | Val: {len(val_files)}")

# --- 3. 增强策略 ---
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityd(keys=["image"]),
    # 大尺寸随机裁剪
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label", spatial_size=roi_size,
        pos=2, neg=1, num_samples=1, image_key="image", image_threshold=0,
    ),
    # 弹性变形 (保持高强度增强)
    Rand3DElasticd(keys=["image", "label"], sigma_range=(5, 8), magnitude_range=(100, 200), prob=0.2, mode=("bilinear", "nearest")),
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

# DataLoader
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=2)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

# --- 4. 模型与权重加载 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | ROI Size: {roi_size}")

model = SwinUNETR(
    img_size=roi_size,
    in_channels=1,
    out_channels=num_classes,
    feature_size=48, 
    use_checkpoint=True, 
).to(device)

# 权重加载逻辑
weight_path = os.path.expanduser("~/projects/Hydro_Seg_Project/model_swinvit.pt")
if os.path.exists(weight_path):
    print(f"Loading pre-trained weights...")
    try:
        checkpoint = torch.load(weight_path)
        pretrained_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model_dict = model.state_dict()
        full_dict = {}
        for k, v in pretrained_dict.items():
            # 暴力修复 Key 不匹配问题
            if k.startswith("module."): target_key = k.replace("module.", "swinViT.", 1)
            else: target_key = "swinViT." + k
            
            if target_key in model_dict:
                if v.shape == model_dict[target_key].shape:
                    full_dict[target_key] = v
        model_dict.update(full_dict)
        model.load_state_dict(model_dict)
        print(f"SUCCESS: Loaded {len(full_dict)} layers from pre-trained weights!")
    except Exception as e: print(f"Error loading weights: {e}")
else:
    print("Warning: Pre-trained weight file not found.")

# --- 5. 策略调整：重点优化脑室和CSF ---
# 标签: 0:背景, 1:CSF, 2:GM, 3:WM, 4:脑室, 5:脉络丛
# 策略: 
#   - CSF(1) & 脑室(4) & 脉络丛(5): 权重 5.0 (核心关注)
#   - GM(2) & WM(3): 权重 0.5 (不重要，容忍错误)
#   - 背景(0): 权重 0.1
class_weights = torch.tensor([0.1, 5.0, 0.5, 0.5, 5.0, 5.0]).to(device)

loss_function = DiceCELoss(
    to_onehot_y=True, 
    softmax=True, 
    ce_weight=class_weights, # 针对性加权
    lambda_dice=1.0, 
    lambda_ce=1.0
)

lv_aux_loss_function = nn.BCEWithLogitsLoss()
alpha_lv = 0.5

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")


# --- 6. 训练循环 (含梯度累积) ---
best_metric = -1
best_metric_epoch = -1
print("Start LARGE ROI Training...")

for epoch in range(max_epochs):
    model.train()
    epoch_loss = 0
    epoch_main_loss = 0
    epoch_lv_loss = 0
    step = 0
    optimizer.zero_grad() # 显式清零
    
    for i, batch_data in enumerate(train_loader):
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        
        outputs = model(inputs)

        # 主任务：原来的 6 类分割 loss
        main_loss = loss_function(outputs, labels)

        # 辅助任务：单独监督 LV(label=4)
        # outputs[:, 4:5, ...] 取出第 4 类（Ventricle）对应的 logits
        lv_logits = outputs[:, 4:5, ...]

        # labels shape 通常是 [B, 1, H, W, D]
        # 构造二值 LV 标签：是 4 的地方为 1，否则为 0
        lv_labels = (labels == 4).float()

        # LV 辅助 loss
        lv_loss = lv_aux_loss_function(lv_logits, lv_labels)

        # 总 loss
        loss = main_loss + alpha_lv * lv_loss
        
        # 梯度累积：Loss 除以累积步数
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        # 每累积一定步数才更新参数
        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
            
        epoch_loss += loss.item() * gradient_accumulation_steps 
        epoch_main_loss += main_loss.item()
        epoch_lv_loss += lv_loss.item()

    epoch_loss /= step
    epoch_main_loss /= step
    epoch_lv_loss /= step

    print(
        f"Epoch {epoch + 1}/{max_epochs}, "
        f"total_loss: {epoch_loss:.4f}, "
        f"main_loss: {epoch_main_loss:.4f}, "
        f"lv_loss: {epoch_lv_loss:.4f}"
    )

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)

                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, 4, model, overlap=0.5
                )
                val_outputs = [
                    AsDiscrete(argmax=True, to_onehot=num_classes)(i)
                    for i in decollate_batch(val_outputs)
                ]
                val_labels = [
                    AsDiscrete(to_onehot=num_classes)(i)
                    for i in decollate_batch(val_labels)
                ]

                dice_metric_batch(y_pred=val_outputs, y=val_labels)

            metric_batch = dice_metric_batch.aggregate()
            mean_metric = torch.mean(metric_batch).item()
            

            # 不含背景时:
            # 0: CSF, 1: GrayMatter, 2: WhiteMatter, 3: Ventricle, 4: ChoroidPlexus
            lv_metric = metric_batch[3].item()

            class_names = ["CSF", "GrayMatter", "WhiteMatter", "Ventricle", "ChoroidPlexus"]
            print(f">> Epoch {epoch + 1} | Mean Dice: {mean_metric:.4f} | LV Dice: {lv_metric:.4f}")
            for i, name in enumerate(class_names):
                print(f"   - {name}: {metric_batch[i].item():.4f}")

            if lv_metric > best_metric:
                best_metric = lv_metric
                best_metric_epoch = epoch + 1
                torch.save(
                    model.state_dict(),
                    os.path.join(log_dir, "best_metric_model_large_lvaux.pth")
                )
                print("   [SAVED] New Best Large Model by LV Dice!")

            dice_metric_batch.reset()

print(f"Training Done. Best LV Dice: {best_metric:.4f} at epoch {best_metric_epoch}")