import os
import argparse
import csv
import torch
import numpy as np

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityd, EnsureTyped, AsDiscrete
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import SwinUNETR, UNet
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--split_file", type=str, required=True)
parser.add_argument("--model_type", choices=["unet", "swin"], required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--roi", type=int, default=96)
parser.add_argument("--output_csv", type=str, default="case_metrics.csv")
parser.add_argument("--overlap", type=float, default=0.5)
parser.add_argument("--sw_mode", type=str, default="constant", choices=["constant", "gaussian"])
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Load split
# -------------------------
with open(args.split_file, "r") as f:
    test_ids = [line.strip() for line in f if line.strip()]

data_dicts = []
for sid in test_ids:
    scan_dir = os.path.join(args.data_dir, sid, "scan_01")
    data_dicts.append({
        "image": os.path.join(scan_dir, "eT1W_FFE_SVR.nii.gz"),
        "label": os.path.join(scan_dir, "segmentation", "gt.nii.gz"),
        "id": sid
    })


# -------------------------
# Transform
# -------------------------
val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")),
    ScaleIntensityd(keys=["image"]),
    EnsureTyped(keys=["image", "label"])
])

val_ds = Dataset(data_dicts, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)


# -------------------------
# Model
# -------------------------
if args.model_type == "swin":
    model = SwinUNETR(
        img_size=(args.roi, args.roi, args.roi),
        in_channels=1,
        out_channels=6,
        feature_size=48
    ).to(device)

elif args.model_type == "unet":
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=6,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="batch",
    ).to(device)

state = torch.load(args.model_path, map_location=device)
model.load_state_dict(state)
model.eval()


# -------------------------
# Global metrics
# -------------------------
mean_dice_metric = DiceMetric(include_background=False, reduction="mean")
lv_dice_metric = DiceMetric(include_background=False, reduction="mean")
lv_hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")


# -------------------------
# CSV
# -------------------------
csv_file = open(args.output_csv, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow([
    "case",
    "lv_dice_raw",
    "lv_hd95_raw"
])


print("Starting raw-only evaluation...")

with torch.no_grad():
    for batch in val_loader:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)
        case_id = batch["id"][0]

        outputs = sliding_window_inference(
            inputs,
            (args.roi, args.roi, args.roi),
            4,
            model,
            overlap=args.overlap,
            mode=args.sw_mode,
        )

        outputs = decollate_batch(outputs)
        labels = decollate_batch(labels)

        outputs_raw = [AsDiscrete(argmax=True, to_onehot=6)(o) for o in outputs]
        labels_list = [AsDiscrete(to_onehot=6)(l) for l in labels]

        # overall mean dice
        mean_dice_metric(y_pred=outputs_raw, y=labels_list)

        # LV only: label = 4
        v_pred = [o[4:5] for o in outputs_raw]
        v_lab = [l[4:5] for l in labels_list]

        case_dice_metric = DiceMetric(include_background=False, reduction="mean")
        case_hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

        case_dice_metric(y_pred=v_pred, y=v_lab)
        case_hd95_metric(y_pred=v_pred, y=v_lab)

        lv_dice_raw = case_dice_metric.aggregate().item()
        lv_hd95_raw = case_hd95_metric.aggregate().item()

        # accumulate global LV metrics
        lv_dice_metric(y_pred=v_pred, y=v_lab)
        lv_hd95_metric(y_pred=v_pred, y=v_lab)

        writer.writerow([
            case_id,
            lv_dice_raw,
            lv_hd95_raw
        ])

        print(f"{case_id}: LV Dice={lv_dice_raw:.4f}, LV HD95={lv_hd95_raw:.4f}")

csv_file.close()

mean_dice = mean_dice_metric.aggregate().item()
lv_dice = lv_dice_metric.aggregate().item()
lv_hd95 = lv_hd95_metric.aggregate().item()

print("-" * 40)
print(f"Mean Dice: {mean_dice:.4f}")
print(f"LV Dice (raw): {lv_dice:.4f}")
print(f"LV HD95 (raw): {lv_hd95:.4f}")
print(f"Saved case-level results to: {args.output_csv}")
print(f"Starting raw-only evaluation... ROI={args.roi}, overlap={args.overlap}")
print(f"Starting raw-only evaluation... ROI={args.roi}, overlap={args.overlap}, mode={args.sw_mode}")
print("-" * 40)