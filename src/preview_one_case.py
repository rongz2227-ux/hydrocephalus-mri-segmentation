import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityd, EnsureTyped
)
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference

def pick_best_slice(mask_3d, axis=2):
    # mask_3d: [H,W,D] if axis=2
    if axis == 0:
        areas = [mask_3d[i,:,:].sum() for i in range(mask_3d.shape[0])]
    elif axis == 1:
        areas = [mask_3d[:,i,:].sum() for i in range(mask_3d.shape[1])]
    else:
        areas = [mask_3d[:,:,i].sum() for i in range(mask_3d.shape[2])]
    return int(np.argmax(areas))

def main():
    data_dir = os.path.expanduser("~/projects/Hydro_Seg_Project/data")
    model_path = os.path.expanduser("~/projects/Hydro_Seg_Project/experiments_swin/best_metric_model_swin.pth")
    sid = "sub_040"  # 你想看哪个就改这个

    img_path = f"{data_dir}/{sid}/scan_01/eT1W_FFE_SVR.nii.gz"
    gt_path  = f"{data_dir}/{sid}/scan_01/segmentation/gt.nii.gz"
    assert os.path.exists(img_path) and os.path.exists(gt_path), "image/gt not found"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ✅ 注意：这是 processed space（RAS + 1mm），Image/GT/Pred 三者一致，所以可视化不会错位
    tfm = Compose([
        LoadImaged(keys=["image","label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        Orientationd(keys=["image","label"], axcodes="RAS"),
        Spacingd(keys=["image","label"], pixdim=(1,1,1), mode=("bilinear","nearest")),
        ScaleIntensityd(keys=["image"]),
        EnsureTyped(keys=["image","label"]),
    ])

    ds = Dataset([{"image": img_path, "label": gt_path}], transform=tfm)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    model = SwinUNETR(
        img_size=(96,96,96),
        in_channels=1,
        out_channels=6,
        feature_size=48,
        use_checkpoint=False,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("model loaded:", model_path)

    lv_id = 4

    with torch.no_grad():
        batch = next(iter(dl))
        x = batch["image"].to(device)   # [1,1,H,W,D]
        y = batch["label"].to(device)   # [1,1,H,W,D]

        logits = sliding_window_inference(x, (96,96,96), 1, model, overlap=0.5)
        pred = torch.argmax(logits, dim=1).cpu().numpy()[0]      # [H,W,D]
        img  = x.cpu().numpy()[0,0]                               # [H,W,D]
        gt   = y.cpu().numpy()[0,0].astype(np.int16)              # [H,W,D]

        gt_lv = (gt == lv_id).astype(np.uint8)
        pr_lv = (pred == lv_id).astype(np.uint8)

        k = pick_best_slice(gt_lv, axis=2)  # axis=2 对应你原脚本的 slice_idx
        print("best LV slice:", k, "gt_lv voxels on slice:", int(gt_lv[:,:,k].sum()))

        outdir = "quick_triplets"
        os.makedirs(outdir, exist_ok=True)

        # 1) 三联图
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1); plt.title("MRI (processed RAS1mm)"); plt.imshow(img[:,:,k], cmap="gray"); plt.axis("off")
        plt.subplot(1,3,2); plt.title("GT"); plt.imshow(gt[:,:,k], cmap="jet", vmin=0, vmax=5); plt.axis("off")
        plt.subplot(1,3,3); plt.title("Pred"); plt.imshow(pred[:,:,k], cmap="jet", vmin=0, vmax=5); plt.axis("off")
        p1 = os.path.join(outdir, f"{sid}_triplet_processed.png")
        plt.savefig(p1, dpi=160, bbox_inches="tight"); plt.close()
        print("[saved]", p1)

        # 2) 边缘/形状更直观：GT红 Pred绿 overlay
        plt.figure(figsize=(6,6))
        plt.imshow(img[:,:,k], cmap="gray"); 
        plt.imshow(np.ma.masked_where(gt_lv[:,:,k]==0, gt_lv[:,:,k]), cmap="Reds", alpha=0.35)
        plt.imshow(np.ma.masked_where(pr_lv[:,:,k]==0, pr_lv[:,:,k]), cmap="Greens", alpha=0.35)
        plt.title(f"{sid} LV overlay (GT red / Pred green), slice={k}")
        plt.axis("off")
        p2 = os.path.join(outdir, f"{sid}_LV_overlay_processed.png")
        plt.savefig(p2, dpi=180, bbox_inches="tight"); plt.close()
        print("[saved]", p2)

if __name__ == "__main__":
    main()