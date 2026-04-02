import os
import sys
import glob
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from custom_losses import BoundaryWeightedDiceLoss

from monai.utils import set_determinism
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityd, RandCropByPosNegLabeld, RandFlipd,
    Rand3DElasticd, EnsureTyped, AsDiscrete
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric

data_dir = os.path.expanduser("~/projects/Hydro_Seg_Project/data")
pretrained_pth = os.path.expanduser("~/projects/Hydro_Seg_Project/experiments_large/best_metric_model_large.pth")
log_dir = os.path.expanduser("~/projects/Hydro_Seg_Project/experiments_boundary_ft_alpha_0.1")
if not os.path.exists(log_dir): os.makedirs(log_dir)

max_epochs = 200
val_interval = 5
batch_size = 1
gradient_accumulation_steps = 4
lr = 1e-5
roi_size = (128, 128, 128)
num_classes = 6
alpha_boundary = 0.1

set_determinism(seed=42)

subjects = sorted(glob.glob(os.path.join(data_dir, "sub_*")))
data_dicts = []
for sub in subjects:
    scan_dirs = sorted(glob.glob(os.path.join(sub, "scan_*")))
    if len(scan_dirs) == 0: continue
    scan_path = scan_dirs[0]
    img_path = os.path.join(scan_path, "eT1W_FFE_SVR.nii.gz")
    lbl_path = os.path.join(scan_path, "segmentation", "gt.nii.gz")
    if os.path.exists(img_path) and os.path.exists(lbl_path):
        data_dicts.append({"image": img_path, "label": lbl_path})

train_files, val_files = data_dicts[:-9], data_dicts[-9:]
print(f"Train: {len(train_files)} | Val: {len(val_files)}")

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityd(keys=["image"]),
    RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=roi_size,
        pos=2, neg=1, num_samples=1, image_key="image", image_threshold=0),
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

train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=2)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = SwinUNETR(img_size=roi_size, in_channels=1, out_channels=num_classes, feature_size=48, use_checkpoint=True).to(device)

if os.path.exists(pretrained_pth):
    model.load_state_dict(torch.load(pretrained_pth, map_location=device))
    print(f"[OK] Loaded weights from: {pretrained_pth}")
else:
    print(f"[WARNING] .pth not found: {pretrained_pth}")

class_weights = torch.tensor([0.1, 5.0, 0.5, 0.5, 5.0, 5.0]).to(device)
global_loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, ce_weight=class_weights, lambda_dice=1.0, lambda_ce=1.0)
boundary_loss_fn = BoundaryWeightedDiceLoss(kernel_size=3, boundary_weight=3.0)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")
hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch")

best_lv_dice = -1
print(f"Start fine-tuning | alpha={alpha_boundary} | lr={lr}")

for epoch in range(max_epochs):
    model.train()
    epoch_loss = 0
    step = 0
    optimizer.zero_grad()

    for i, batch_data in enumerate(train_loader):
        step += 1
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        outputs = model(inputs)
        loss_global = global_loss_fn(outputs, labels)
        lv_logits = outputs[:, 4:5, ...]
        lv_targets = (labels == 4).float()
        loss_boundary = boundary_loss_fn(lv_logits, lv_targets)
        total_loss = loss_global + alpha_boundary * loss_boundary

        (total_loss / gradient_accumulation_steps).backward()
        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += total_loss.item()

    epoch_loss /= step
    print(f"Epoch {epoch+1}/{max_epochs} | loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)
                val_outputs = sliding_window_inference(val_inputs, roi_size, 4, model, overlap=0.25, mode="constant")
                val_out_disc = [AsDiscrete(argmax=True, to_onehot=num_classes)(i) for i in decollate_batch(val_outputs)]
                val_lbl_disc = [AsDiscrete(to_onehot=num_classes)(i) for i in decollate_batch(val_labels)]
                dice_metric_batch(y_pred=val_out_disc, y=val_lbl_disc)
                hd_metric(y_pred=val_out_disc, y=val_lbl_disc)

        metric_batch = dice_metric_batch.aggregate()
        hd_batch = hd_metric.aggregate()
        lv_dice = metric_batch[3].item()
        lv_hd95 = hd_batch[3].item()
        mean_dice = torch.mean(metric_batch).item()

        print(f">> Epoch {epoch+1} | Mean Dice: {mean_dice:.4f} | LV Dice: {lv_dice:.4f} | LV HD95: {lv_hd95:.4f}")

        if lv_dice > best_lv_dice:
            best_lv_dice = lv_dice
            torch.save(model.state_dict(), os.path.join(log_dir, "best_boundary_finetune.pth"))
            print(f"   [SAVED] LV Dice={lv_dice:.4f}, HD95={lv_hd95:.4f}")

        dice_metric_batch.reset()
        hd_metric.reset()

print(f"Done. Best LV Dice: {best_lv_dice:.4f}")
