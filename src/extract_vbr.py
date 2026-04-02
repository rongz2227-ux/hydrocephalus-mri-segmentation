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

records = []
for pred_dir, split in dirs.items():
    paths = sorted(glob.glob(os.path.join(pred_dir, "*_pred_on_raw.nii.gz")))
    for p in paths:
        sub_id = os.path.basename(p).replace("_pred_on_raw.nii.gz", "")
        nii = nib.load(p)
        data = nii.get_fdata().astype(np.int32)
        
        # 计算体素数量
        lv_voxels = np.sum(data == 4)
        brain_voxels = np.sum(data > 0)  # 1-5均为脑内结构
        
        # 计算占比
        vbr = lv_voxels / brain_voxels if brain_voxels > 0 else 0
        
        records.append({
            "subject_id": sub_id,
            "vbr_ratio": round(vbr, 4)
        })
        print(f"{sub_id}: VBR = {vbr:.4f}")

# 与现有完整数据合并
vbr_df = pd.DataFrame(records)
full_df = pd.read_csv("results/lv_full.csv")
final_df = full_df.merge(vbr_df, on="subject_id")
final_df.to_csv("results/lv_final_features.csv", index=False)
print(f"\n保存完成：results/lv_final_features.csv，共 {len(final_df)} 个病例")
