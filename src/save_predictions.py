import os
import glob
import argparse
import torch
import numpy as np
import nibabel as nib

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityd, EnsureTyped, AsDiscrete
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--split_file", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--roi", type=int, default=128)
parser.add_argument("--overlap", type=float, default=0.25)
parser.add_argument("--sw_mode", type=str, default="constant")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(args.split_file) as f:
    test_ids = [l.strip() for l in f if l.strip()]

data_dicts = []
for sid in test_ids:
    scan_dir = os.path.join(args.data_dir, sid, "scan_01")
    data_dicts.append({
        "image": os.path.join(scan_dir, "eT1W_FFE_SVR.nii.gz"),
        "label": os.path.join(scan_dir, "segmentation", "gt.nii.gz"),
        "id": sid
    })

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

model = SwinUNETR(
    img_size=(args.roi, args.roi, args.roi),
    in_channels=1, out_channels=6, feature_size=48
).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

with torch.no_grad():
    for batch in val_loader:
        inputs = batch["image"].to(device)
        sid = batch["id"][0]

        # 取原始 NIfTI 的 affine（用于保存时保持空间信息一致）
        orig_nii = nib.load(batch["label_meta_dict"]["filename_or_obj"][0] 
                            if "label_meta_dict" in batch 
                            else data_dicts[val_loader.dataset.data.index(
                                next(d for d in data_dicts if d["id"] == sid))]["label"])

        outputs = sliding_window_inference(
            inputs, (args.roi, args.roi, args.roi), 4, model,
            overlap=args.overlap, mode=args.sw_mode
        )

        pred = outputs[0].argmax(dim=0).cpu().numpy()  # [D, H, W]
        lv_pred = (pred == 4).astype(np.uint8)          # LV 二值 mask

        # 用 label 的 affine 保持几何一致
        ref_nii = nib.load(data_dicts[[d["id"] for d in data_dicts].index(sid)]["label"])
        out_nii = nib.Nifti1Image(lv_pred, ref_nii.affine, ref_nii.header)
        nib.save(out_nii, os.path.join(args.output_dir, f"{sid}_pred_lv.nii.gz"))
        print(f"[SAVED] {sid}_pred_lv.nii.gz")