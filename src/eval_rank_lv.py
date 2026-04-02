import os, glob, argparse
import torch
import numpy as np

import torch.nn.functional as F

def binary_closing_3d(mask_3d: torch.Tensor, k: int = 3, iters: int = 1) -> torch.Tensor:
    """
    mask_3d: torch tensor [H, W, D] or [D, H, W] (any 3D shape is fine as long as it's 3D)
    Returns: same shape, float tensor {0,1} after closing (dilation then erosion)
    Implemented via max_pool3d.
    """
    assert mask_3d.ndim == 3
    x = (mask_3d > 0.5).float().unsqueeze(0).unsqueeze(0)  # [1,1,*,*,*]
    pad = k // 2

    for _ in range(iters):
        # dilation
        x = F.max_pool3d(x, kernel_size=k, stride=1, padding=pad)
        # erosion: min_pool = -max_pool(-x)
        x = -F.max_pool3d(-x, kernel_size=k, stride=1, padding=pad)

    return x.squeeze(0).squeeze(0)

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityd, EnsureTyped, AsDiscrete, KeepLargestConnectedComponent
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
# from monai.transforms import Closing

from monai.networks.nets import SwinUNETR
from monai.networks.nets import BasicUNet  # 用于 experiments_opt / experiments
from monai.metrics import HausdorffDistanceMetric

def is_swin_ckpt(sd_keys):
    return any(k.startswith("swinViT.") for k in sd_keys)


def swin_in_channels(sd):
    return int(sd["swinViT.patch_embed.proj.weight"].shape[1])


def build_model_from_ckpt(ckpt_path, device):
    sd = torch.load(ckpt_path, map_location="cpu")
    keys = list(sd.keys())

    if is_swin_ckpt(keys):
        in_ch = swin_in_channels(sd)
        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=in_ch,
            out_channels=6,
            feature_size=48,
            use_checkpoint=False,
        ).to(device)
        model.load_state_dict(sd)
        model.eval()
        return model, "swin", in_ch

    # UNet 系列（你的 ckpt key 是 model.0.conv...，很像 BasicUNet 的递归结构）
    # 这里按你的 ckpt 第一层 weight (16,1,3,3,3) 推断 features=16
    # out_channels：你项目 gt 是 0-4，但你训练可能用 6；为了兼容，我们按训练脚本的 out_channels 去定义。
    # 如果加载报错，我们再把 out_channels 调整为 5。
    model = BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=6,
        features=(16, 32, 64, 128, 256, 32),
    ).to(device)

    try:
        model.load_state_dict(sd)
    except RuntimeError:
        # fallback：有些 run 可能 out_channels=5
        model = BasicUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=5,
            features=(16, 32, 64, 128, 256, 32),
        ).to(device)
        model.load_state_dict(sd)

    model.eval()
    return model, "unet", 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", nargs="+", required=True, help="list of ckpt paths")
    parser.add_argument("--data_dir", type=str, default="~/projects/Hydro_Seg_Project/data")
    parser.add_argument("--val_last_n", type=int, default=9)
    parser.add_argument("--roi", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--sw_batch_size", type=int, default=4)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--close_k", type=int, default=0, help="closing kernel size for LV (0 disables), e.g., 3 or 5")
    parser.add_argument("--close_iters", type=int, default=1, help="closing iterations")
    parser.add_argument("--split_file", type=str, default=None,
                        help="path to subjects list, e.g. splits/test_subjects.txt")
    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- prepare file list ----
    subjects = sorted(glob.glob(os.path.join(data_dir, "sub_*")))
    data_dicts = []
    for sub in subjects:
        scan_dirs = sorted(glob.glob(os.path.join(sub, "scan_*")))
        if len(scan_dirs) == 0:
            continue
        data_dicts.append({
            "image": os.path.join(scan_dirs[0], "eT1W_FFE_SVR.nii.gz"),
            "label": os.path.join(scan_dirs[0], "segmentation", "gt.nii.gz")
        })

    def sub_id_from_image(p):
        # .../data/sub_001/scan_01/eT1W...nii.gz -> sub_001
        return os.path.basename(os.path.dirname(os.path.dirname(p)))

    if args.split_file:
        split_path = os.path.expanduser(args.split_file)
        with open(split_path, "r") as f:
            keep = set(line.strip() for line in f if line.strip())
        val_files = [d for d in data_dicts if sub_id_from_image(d["image"]) in keep]
        print(f"[INFO] Using split_file={split_path} | cases={len(val_files)}")
    else:
        val_files = data_dicts[-args.val_last_n:]
        print(f"[INFO] Using val_last_n={args.val_last_n} | cases={len(val_files)}")

    # ---- transforms / loader ----
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityd(keys=["image"]),
        EnsureTyped(keys=["image", "label"]),
    ])
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # safer post: only keep LCC for ventricle (label=4) + optional closing
    post_lcc = KeepLargestConnectedComponent(applied_labels=[4])
    close_k = int(args.close_k)
    close_iters = int(args.close_iters)
    results = []

    for ckpt in args.model_paths:
        ckpt = os.path.expanduser(ckpt)
        model, arch, in_ch = build_model_from_ckpt(ckpt, device)

        dice_mean_raw = DiceMetric(include_background=False, reduction="mean")
        dice_mean_post = DiceMetric(include_background=False, reduction="mean")
        dice_v_raw = DiceMetric(include_background=False, reduction="mean")
        dice_v_post = DiceMetric(include_background=False, reduction="mean")
        hd95_v_raw = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
        hd95_v_post = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

        with torch.no_grad():
            for data in val_loader:
                x = data["image"].to(device)  # [B,1,H,W,D]
                y = data["label"].to(device)

                if in_ch == 3 and x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1, 1)

                logits = sliding_window_inference(
                    x, tuple(args.roi), args.sw_batch_size, model, overlap=args.overlap
                )

                out_raw = [AsDiscrete(argmax=True, to_onehot=6)(i) for i in decollate_batch(logits)]
                lab = [AsDiscrete(to_onehot=6)(i) for i in decollate_batch(y)]

                dice_mean_raw(y_pred=out_raw, y=lab)

                v_raw = [i[4:5] for i in out_raw]
                v_lab = [i[4:5] for i in lab]
                dice_v_raw(y_pred=v_raw, y=v_lab)

                out_post = []
                for onehot in out_raw:
                    x = post_lcc(onehot)  # LCC on LV
                    if close_k > 0:
                        x4 = binary_closing_3d(x[4], k=close_k, iters=close_iters)
                        x[4] = (x4 > 0.5).float()
                    out_post.append(x)

                dice_mean_post(y_pred=out_post, y=lab)

                v_post = [i[4:5] for i in out_post]
                dice_v_post(y_pred=v_post, y=v_lab)

                hd95_v_raw(y_pred=v_raw, y=v_lab)
                hd95_v_post(y_pred=v_post, y=v_lab)

        mean_raw = dice_mean_raw.aggregate().item()
        mean_post = dice_mean_post.aggregate().item()
        vraw = dice_v_raw.aggregate().item()
        vpost = dice_v_post.aggregate().item()
        hd_raw = hd95_v_raw.aggregate().item()
        hd_post = hd95_v_post.aggregate().item()

        results.append((ckpt, arch, in_ch, mean_raw, mean_post, vraw, vpost))

        print(f"\n=== {ckpt} ===")
        print(f"arch={arch} in_ch={in_ch}")
        print(f"Mean Dice raw/post: {mean_raw:.4f} / {mean_post:.4f}")
        print(f"Ventricle Dice raw/post: {vraw:.4f} / {vpost:.4f}")
        print(f"Ventricle HD95 raw/post: {hd_raw:.4f} / {hd_post:.4f}")

    results.sort(key=lambda x: x[6], reverse=True)
    print("\n=== LEADERBOARD (by Post Ventricle Dice, label=4) ===")
    for ckpt, arch, in_ch, mean_raw, mean_post, vraw, vpost in results:
        print(f"{vpost:.4f} | raw={vraw:.4f} | mean_post={mean_post:.4f} | {arch} in_ch={in_ch} | {ckpt}")

if __name__ == "__main__":
    main()