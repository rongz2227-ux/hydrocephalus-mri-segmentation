import os
import glob
import argparse
import numpy as np
import nibabel as nib
import csv
from scipy.spatial.distance import cdist
from scipy import ndimage


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


def extract_surface(mask: np.ndarray) -> np.ndarray:
    """
    Return coordinates of surface voxels.
    """
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing *_pred_lv_lcc.nii.gz")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory containing *_gt_processed.nii.gz")
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    pred_paths = sorted(glob.glob(os.path.join(args.pred_dir, "*_pred_lv_lcc.nii.gz")))
    print(f"Found {len(pred_paths)} LCC prediction files.")

    rows = []
    dices = []
    hd95s = []

    for pred_path in pred_paths:
        sid = os.path.basename(pred_path).replace("_pred_lv_lcc.nii.gz", "")
        gt_path = os.path.join(args.gt_dir, f"{sid}_gt_processed.nii.gz")

        if not os.path.exists(gt_path):
            print(f"Skipping {sid}: GT not found.")
            continue

        pred = nib.load(pred_path).get_fdata()
        pred = (pred > 0).astype(np.uint8)

        gt_all = nib.load(gt_path).get_fdata().astype(np.uint8)
        gt_lv = (gt_all == 4).astype(np.uint8)

        d = dice_score(pred, gt_lv)
        h = hd95(pred, gt_lv, spacing=(1.0, 1.0, 1.0))

        dices.append(d)
        if not np.isnan(h):
            hd95s.append(h)

        rows.append([sid, d, h])

        print(f"{sid}: LV Dice={d:.4f}, LV HD95={h:.4f}")

    mean_dice = np.mean(dices) if len(dices) > 0 else np.nan
    mean_hd95 = np.mean(hd95s) if len(hd95s) > 0 else np.nan

    print("-" * 40)
    print(f"LV Dice (LCC): {mean_dice:.4f}")
    print(f"LV HD95 (LCC): {mean_hd95:.4f}")

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "lv_dice_lcc", "lv_hd95_lcc"])
        writer.writerows(rows)
        writer.writerow([])
        writer.writerow(["MEAN", mean_dice, mean_hd95])

    print(f"Saved case-level results to: {args.output_csv}")


if __name__ == "__main__":
    main()