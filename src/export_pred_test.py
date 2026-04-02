import os, glob, argparse
import numpy as np
import torch
import nibabel as nib
from nibabel.processing import resample_from_to

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityd, EnsureTyped, AsDiscrete, KeepLargestConnectedComponent
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--data_dir", default="~/projects/Hydro_Seg_Project/data")
    ap.add_argument("--out_dir", default="preds_test_large_on_raw")
    ap.add_argument("--roi", type=int, nargs=3, default=[96,96,96])
    ap.add_argument("--sw_batch_size", type=int, default=1)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--lcc", action="store_true", help="apply LCC to LV (label=4) in processed space")
    args = ap.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.expanduser(args.split_file), "r") as f:
        keep = set(line.strip() for line in f if line.strip())

    # build file list (keep raw paths for resample target)
    subjects = sorted(glob.glob(os.path.join(data_dir, "sub_*")))
    data_dicts = []
    for sub in subjects:
        sid = os.path.basename(sub)
        if sid not in keep:
            continue
        scan_dirs = sorted(glob.glob(os.path.join(sub, "scan_*")))
        if not scan_dirs:
            continue
        raw_img = os.path.join(scan_dirs[0], "eT1W_FFE_SVR.nii.gz")
        raw_gt  = os.path.join(scan_dirs[0], "segmentation", "gt.nii.gz")
        if os.path.exists(raw_img) and os.path.exists(raw_gt):
            data_dicts.append({"sid": sid, "image": raw_img, "label": raw_gt})

    print("[INFO] cases:", len(data_dicts))

    # IMPORTANT: model runs on processed space (RAS + 1mm)
    proc_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0,1.0,1.0), mode=("bilinear","nearest")),
        ScaleIntensityd(keys=["image"]),
        EnsureTyped(keys=["image", "label"]),
    ])

    ds = Dataset(data=data_dicts, transform=proc_transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNETR(img_size=tuple(args.roi), in_channels=1, out_channels=6,
                     feature_size=48, use_checkpoint=False).to(device)
    sd = torch.load(os.path.expanduser(args.model_path), map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    post_lcc = KeepLargestConnectedComponent(applied_labels=[4]) if args.lcc else None

    with torch.no_grad():
        for batch in loader:
            sid = batch["sid"][0]
            x = batch["image"].to(device)

            # infer in processed space
            logits = sliding_window_inference(x, tuple(args.roi), args.sw_batch_size, model, overlap=args.overlap)
            onehot = [AsDiscrete(argmax=True, to_onehot=6)(i) for i in decollate_batch(logits)][0]  # [6,H,W,D]
            if post_lcc is not None:
                onehot = post_lcc(onehot)

            pred_proc = torch.argmax(onehot, dim=0).cpu().numpy().astype(np.uint8)  # processed labelmap [H,W,D]

            # save processed pred temporarily with correct processed affine
            proc_aff = batch["image_meta_dict"]["affine"][0].cpu().numpy()
            pred_proc_nii = nib.Nifti1Image(pred_proc, proc_aff)

            # resample to RAW grid (target is raw image file)
            raw_img_path = batch["image_meta_dict"]["filename_or_obj"][0]
            raw_img_nii = nib.load(raw_img_path)   # raw image has desired shape/affine
            pred_raw_nii = resample_from_to(pred_proc_nii, raw_img_nii, order=0)  # nearest

            out_path = os.path.join(out_dir, f"{sid}_pred_on_raw.nii.gz")
            nib.save(pred_raw_nii, out_path)
            print("[SAVED]", out_path, "shape:", pred_raw_nii.shape)
            
if __name__ == "__main__":
    main()
