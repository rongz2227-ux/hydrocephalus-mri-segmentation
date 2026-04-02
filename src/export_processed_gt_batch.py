import os
import glob
import argparse
import nibabel as nib
import numpy as np

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    EnsureTyped,
)
from monai.data import Dataset, DataLoader


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
        gt_path = os.path.join(scan_path, "segmentation", "gt.nii.gz")

        if os.path.exists(gt_path):
            items.append({"label": gt_path, "id": sid})
        else:
            print(f"Skipping {sid}: missing gt.")
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    subject_ids = read_subjects(args.split_file)
    data_dicts = build_data_dicts(args.data_dir, subject_ids)
    print(f"Found {len(data_dicts)} GT files.")

    transforms = Compose([
        LoadImaged(keys=["label"]),
        EnsureChannelFirstd(keys=["label"]),
        Orientationd(keys=["label"], axcodes="RAS"),
        Spacingd(keys=["label"], pixdim=(1.0, 1.0, 1.0), mode=("nearest",)),
        EnsureTyped(keys=["label"]),
    ])

    ds = Dataset(data=data_dicts, transform=transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    for batch in loader:
        sid = batch["id"][0]
        label = batch["label"][0, 0].cpu().numpy().astype(np.uint8)

        # 这里只导出 processed GT 全标签
        out_path = os.path.join(args.output_dir, f"{sid}_gt_processed.nii.gz")
        nii = nib.Nifti1Image(label, np.eye(4))
        nib.save(nii, out_path)
        print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()