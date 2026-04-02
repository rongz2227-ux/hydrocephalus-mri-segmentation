import os
import glob
import argparse
import torch
import numpy as np

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityd,
    EnsureTyped,
    AsDiscrete,
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference

import nibabel as nib


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
        lbl_path = os.path.join(scan_path, "segmentation", "gt.nii.gz")  # 可有可无，这里只是为了保持结构清晰

        if os.path.exists(img_path):
            items.append({
                "image": img_path,
                "id": sid,
            })
        else:
            print(f"Skipping {sid}: missing image.")
    return items


def save_nifti(array_3d, out_path):
    """
    保存 processed space 下的 NIfTI。
    这里用单位仿射矩阵保存。
    因为当前目标是先在 processed RAS 1mm space 下做后续分析。
    """
    affine = np.eye(4)
    nii = nib.Nifti1Image(array_3d.astype(np.uint8), affine)
    nib.save(nii, out_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--roi", type=int, default=128)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--num_classes", type=int, default=6)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"ROI={args.roi}, overlap={args.overlap}")

    # 1. 读取 subject 列表
    subject_ids = read_subjects(args.split_file)
    print(f"Subjects to export: {len(subject_ids)}")

    # 2. 构造数据
    data_dicts = build_data_dicts(args.data_dir, subject_ids)
    print(f"Valid subjects found: {len(data_dicts)}")

    # 3. 预处理（与评估一致）
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear",)),
        ScaleIntensityd(keys=["image"]),
        EnsureTyped(keys=["image"]),
    ])

    ds = Dataset(data=data_dicts, transform=transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # 4. 模型
    model = SwinUNETR(
        img_size=(args.roi, args.roi, args.roi),
        in_channels=1,
        out_channels=args.num_classes,
        feature_size=48,
    ).to(device)

    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 5. 推理并保存
    with torch.no_grad():
        for batch in loader:
            torch.cuda.empty_cache()
            inputs = batch["image"].to(device)
            sid = batch["id"][0]

            outputs = sliding_window_inference(
                inputs,
                (args.roi, args.roi, args.roi),
                1,
                model,
                overlap=args.overlap,
            )

            # 转成 one-hot
            outputs_list = decollate_batch(outputs)
            outputs_discrete = [
                AsDiscrete(argmax=True, to_onehot=args.num_classes)(o)
                for o in outputs_list
            ]

            # 当前 batch_size=1，只取第一个
            pred_onehot = outputs_discrete[0].cpu().numpy()   # shape: [C, H, W, D]

            # multiclass: argmax over channel
            pred_multiclass = np.argmax(pred_onehot, axis=0).astype(np.uint8)

            # LV binary: label == 4
            pred_lv = (pred_multiclass == 4).astype(np.uint8)

            # 保存
            out_multiclass = os.path.join(args.output_dir, f"{sid}_pred_multiclass.nii.gz")
            out_lv = os.path.join(args.output_dir, f"{sid}_pred_lv.nii.gz")

            save_nifti(pred_multiclass, out_multiclass)
            save_nifti(pred_lv, out_lv)

            print(f"[SAVED] {sid}")
            print(f"  multiclass -> {out_multiclass}")
            print(f"  lv mask    -> {out_lv}")

    print("Export finished.")


if __name__ == "__main__":
    main()