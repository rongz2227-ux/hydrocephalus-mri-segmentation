import os
import glob
import argparse
import numpy as np
import nibabel as nib
import torch

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityd, EnsureTyped, AsDiscrete
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import SwinUNETR, UNet
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from scipy.spatial.distance import cdist
from scipy import ndimage
import csv


def read_subjects(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def build_data_dicts(data_dir, subject_ids):
    items = []
    for sid in subject_ids:
        sub_dir = os.path.join(data_dir, sid)
        scan_dirs = sorted(glob.glob(os.path.join(sub_dir, "scan_*")))
        if len(scan_dirs) == 0:
            print(f"Skipping {sid}: no scan_* folder.")
            continue

        scan_path = scan_dirs[0]
        img_path = os.path.join(scan_path, "eT1W_FFE_SVR.nii.gz")
        lbl_path = os.path.join(scan_path, "segmentation", "gt.nii.gz")

        if os.path.exists(img_path) and os.path.exists(lbl_path):
            items.append({"image": img_path, "label": lbl_path, "id": sid})
        else:
            print(f"Skipping {sid}: missing image or label.")
    return items


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


def extract_surface(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return np.zeros((0, 3), dtype=np.int32)
    eroded = ndimage.binary_erosion(mask)
    surface = mask.astype(bool) ^ eroded.astype(bool)
    coords = np.argwhere(surface)
    return coords


def hd95(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan

    pred_surf = extract_surface(pred)
    gt_surf = extract_surface(gt)

    if len(pred_surf) == 0 or len(gt_surf) == 0:
        return np.nan

    pred_surf = pred_surf * np.array(spacing)
    gt_surf = gt_surf * np.array(spacing)

    dists_pred_to_gt = cdist(pred_surf, gt_surf).min(axis=1)
    dists_gt_to_pred = cdist(gt_surf, pred_surf).min(axis=1)

    all_dists = np.concatenate([dists_pred_to_gt, dists_gt_to_pred])
    return np.percentile(all_dists, 95)


def infer_once(model, inputs, roi_size, overlap):
    outputs = sliding_window_inference(
        inputs,
        roi_size,
        1,   # 更稳，省显存
        model,
        overlap=overlap,
        mode="constant",
    )
    return outputs


def infer_with_flip_tta(model, inputs, roi_size, overlap):
    """
    inputs shape: [B, C, H, W, D]
    We do:
      - original
      - flip H axis
      - flip W axis
      - flip D axis
    Then flip predictions back and average logits.
    """
    preds = []

    # 1) original
    out = infer_once(model, inputs, roi_size, overlap)
    preds.append(out)

    # 2) flip axis=2 (H)
    inp_flip_h = torch.flip(inputs, dims=[2])
    out_flip_h = infer_once(model, inp_flip_h, roi_size, overlap)
    out_flip_h = torch.flip(out_flip_h, dims=[2])
    preds.append(out_flip_h)

    # 3) flip axis=3 (W)
    inp_flip_w = torch.flip(inputs, dims=[3])
    out_flip_w = infer_once(model, inp_flip_w, roi_size, overlap)
    out_flip_w = torch.flip(out_flip_w, dims=[3])
    preds.append(out_flip_w)

    # 4) flip axis=4 (D)
    inp_flip_d = torch.flip(inputs, dims=[4])
    out_flip_d = infer_once(model, inp_flip_d, roi_size, overlap)
    out_flip_d = torch.flip(out_flip_d, dims=[4])
    preds.append(out_flip_d)

    # average logits
    avg_pred = torch.mean(torch.stack(preds, dim=0), dim=0)
    return avg_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True, choices=["swin", "unet"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--roi", type=int, default=128)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    model_path = os.path.expanduser(args.model_path)
    split_file = os.path.expanduser(args.split_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Starting TTA evaluation... ROI={args.roi}, overlap={args.overlap}")

    subject_ids = read_subjects(split_file)
    data_dicts = build_data_dicts(data_dir, subject_ids)

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityd(keys=["image"]),
        EnsureTyped(keys=["image", "label"]),
    ])

    ds = Dataset(data=data_dicts, transform=transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    num_classes = 6

    if args.model_type == "swin":
        model = SwinUNETR(
            img_size=(args.roi, args.roi, args.roi),
            in_channels=1,
            out_channels=num_classes,
            feature_size=48,
            use_checkpoint=True,
        ).to(device)
    else:
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
        ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    rows = []
    mean_dice_list = []
    lv_dice_list = []
    lv_hd95_list = []

    with torch.no_grad():
        for batch in loader:
            sid = batch["id"][0]
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = infer_with_flip_tta(
                model=model,
                inputs=inputs,
                roi_size=(args.roi, args.roi, args.roi),
                overlap=args.overlap,
            )

            outputs_discrete = [AsDiscrete(argmax=True, to_onehot=num_classes)(o) for o in decollate_batch(outputs)]
            labels_discrete = [AsDiscrete(to_onehot=num_classes)(o) for o in decollate_batch(labels)]

            # mean dice over all foreground classes
            dice_metric = DiceMetric(include_background=False, reduction="mean")
            dice_metric(y_pred=outputs_discrete, y=labels_discrete)
            case_mean_dice = dice_metric.aggregate().item()
            dice_metric.reset()

            # LV only: class index 4 in original labels -> index 4 in onehot (with background channel included)
            pred_lv = outputs_discrete[0][4].cpu().numpy().astype(np.uint8)
            gt_lv = labels_discrete[0][4].cpu().numpy().astype(np.uint8)

            case_lv_dice = dice_score(pred_lv, gt_lv)
            case_lv_hd95 = hd95(pred_lv, gt_lv, spacing=(1.0, 1.0, 1.0))

            mean_dice_list.append(case_mean_dice)
            lv_dice_list.append(case_lv_dice)
            if not np.isnan(case_lv_hd95):
                lv_hd95_list.append(case_lv_hd95)

            rows.append([sid, case_mean_dice, case_lv_dice, case_lv_hd95])

            print(f"{sid}: LV Dice={case_lv_dice:.4f}, LV HD95={case_lv_hd95:.4f}")

    mean_dice = np.mean(mean_dice_list) if len(mean_dice_list) > 0 else np.nan
    lv_dice = np.mean(lv_dice_list) if len(lv_dice_list) > 0 else np.nan
    lv_hd95 = np.mean(lv_hd95_list) if len(lv_hd95_list) > 0 else np.nan

    print("-" * 40)
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"LV Dice (TTA): {lv_dice:.4f}")
    print(f"LV HD95 (TTA): {lv_hd95:.4f}")

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "mean_dice", "lv_dice", "lv_hd95"])
        writer.writerows(rows)
        writer.writerow([])
        writer.writerow(["MEAN", mean_dice, lv_dice, lv_hd95])

    print(f"Saved case-level results to: {args.output_csv}")


if __name__ == "__main__":
    main()