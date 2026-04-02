import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd

dirs = {
    "preds_all_large_train": "train",
    "preds_all_large_val":   "val",
    "preds_all_large_test":  "test",
}

LV_LABEL = 4  # 确认这个编号和你的数据一致

records = []
for pred_dir, split in dirs.items():
    paths = sorted(glob.glob(os.path.join(pred_dir, "*_pred_on_raw.nii.gz")))
    for p in paths:
        sub_id = os.path.basename(p).replace("_pred_on_raw.nii.gz", "")
        nii = nib.load(p)
        data = nii.get_fdata().astype(np.int32)
        zooms = nii.header.get_zooms()
        voxel_vol_mm3 = zooms[0] * zooms[1] * zooms[2]

        lv_mask = (data == LV_LABEL)
        lv_vol_mm3 = lv_mask.sum() * voxel_vol_mm3
        lv_vol_cm3 = lv_vol_mm3 / 1000.0

        records.append({
            "subject_id": sub_id,
            "split": split,
            "lv_volume_mm3": lv_vol_mm3,
            "lv_volume_cm3": round(lv_vol_cm3, 4),
        })
        print(f"{sub_id}: LV = {lv_vol_cm3:.2f} cm³")

df = pd.DataFrame(records)
os.makedirs("results", exist_ok=True)
df.to_csv("results/lv_features.csv", index=False)
print(f"\n保存完成：results/lv_features.csv，共 {len(df)} 个病例")
