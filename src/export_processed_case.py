import os
import glob
import argparse
import numpy as np
import nibabel as nib

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityd,
    EnsureTyped,
)
from monai.data import Dataset, DataLoader


def save_nifti(array_3d, out_path, dtype=np.float32):
    affine = np.eye(4)
    nii = nib.Nifti1Image(array_3d.astype(dtype), affine)
    nib.save(nii, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--subject_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sub_dir = os.path.join(args.data_dir, args.subject_id)
    scan_dirs = sorted(glob.glob(os.path.join(sub_dir, "scan_*")))
    if len(scan_dirs) == 0:
        raise FileNotFoundError(f"No scan_* folder found for {args.subject_id}")

    scan_path = scan_dirs[0]
    img_path = os.path.join(scan_path, "eT1W_FFE_SVR.nii.gz")
    gt_path = os.path.join(scan_path, "segmentation", "gt.nii.gz")

    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)
    if not os.path.exists(gt_path):
        raise FileNotFoundError(gt_path)

    data = [{"image": img_path, "label": gt_path}]

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityd(keys=["image"]),
        EnsureTyped(keys=["image", "label"]),
    ])

    ds = Dataset(data=data, transform=transforms)
    loader = DataLoader(ds, batch_size=1)

    batch = next(iter(loader))
    image = batch["image"][0, 0].cpu().numpy()   # [H, W, D]
    label = batch["label"][0, 0].cpu().numpy()   # [H, W, D]

    out_img = os.path.join(args.output_dir, f"{args.subject_id}_image_processed.nii.gz")
    out_gt = os.path.join(args.output_dir, f"{args.subject_id}_gt_processed.nii.gz")

    save_nifti(image, out_img, dtype=np.float32)
    save_nifti(label, out_gt, dtype=np.uint8)

    print(f"[SAVED] processed image -> {out_img}")
    print(f"[SAVED] processed gt    -> {out_gt}")
    print(f"Processed image shape: {image.shape}")
    print(f"Processed gt shape:    {label.shape}")


if __name__ == "__main__":
    main()