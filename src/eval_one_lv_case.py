import nibabel as nib
import numpy as np
from scipy.spatial.distance import cdist
from scipy import ndimage


gt_path = "./postprocess_eval/test_gt_lv/sub_013_gt_lv.nii.gz"
pred_before_path = "./preds_lv_oldlarge_overlap025_test/sub_013_pred_lv.nii.gz"
pred_after_path = "./postprocess_eval/test_pred_lv_cc2/sub_013_pred_lv_cc2.nii.gz"


def dice_score(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    inter = np.logical_and(gt, pred).sum()
    denom = gt.sum() + pred.sum()

    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


def get_surface_points(mask):
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.empty((0, 3), dtype=np.int32)

    eroded = ndimage.binary_erosion(mask)
    surface = np.logical_and(mask, np.logical_not(eroded))
    pts = np.argwhere(surface)
    return pts


def hd95(gt, pred, spacing=(1.0, 1.0, 1.0)):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    if gt.sum() == 0 and pred.sum() == 0:
        return 0.0
    if gt.sum() == 0 or pred.sum() == 0:
        return np.nan

    gt_pts = get_surface_points(gt)
    pred_pts = get_surface_points(pred)

    if len(gt_pts) == 0 or len(pred_pts) == 0:
        return np.nan

    gt_pts = gt_pts * np.array(spacing)
    pred_pts = pred_pts * np.array(spacing)

    dists_gt_to_pred = cdist(gt_pts, pred_pts).min(axis=1)
    dists_pred_to_gt = cdist(pred_pts, gt_pts).min(axis=1)

    all_dists = np.concatenate([dists_gt_to_pred, dists_pred_to_gt])
    return np.percentile(all_dists, 95)


gt_nii = nib.load(gt_path)
pred_before_nii = nib.load(pred_before_path)
pred_after_nii = nib.load(pred_after_path)

gt = (gt_nii.get_fdata() > 0).astype(np.uint8)
pred_before = (pred_before_nii.get_fdata() > 0).astype(np.uint8)
pred_after = (pred_after_nii.get_fdata() > 0).astype(np.uint8)

spacing = gt_nii.header.get_zooms()[:3]

dice_before = dice_score(gt, pred_before)
hd95_before = hd95(gt, pred_before, spacing=spacing)

dice_after = dice_score(gt, pred_after)
hd95_after = hd95(gt, pred_after, spacing=spacing)

print("=== sub_013 ===")
print(f"GT voxels: {int(gt.sum())}")
print(f"Pred BEFORE voxels: {int(pred_before.sum())}")
print(f"Pred AFTER  voxels: {int(pred_after.sum())}")
print()
print(f"Dice BEFORE: {dice_before:.4f}")
print(f"HD95 BEFORE: {hd95_before:.4f}")
print(f"Dice AFTER : {dice_after:.4f}")
print(f"HD95 AFTER : {hd95_after:.4f}")