import os
import glob
import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm
from monai.transforms import Resize

# --- 配置 ---
data_dir = os.path.expanduser("~/projects/Hydro_Seg_Project/data")
save_path = os.path.expanduser("~/projects/Hydro_Seg_Project/data/probability_map.nii.gz")

# 设定一个标准尺寸 (Standard Shape)
# 256x256x256 是医学影像中非常通用的标准尺寸
target_shape = (256, 256, 256)

# 1. 获取所有训练数据的标签路径
subjects = sorted(glob.glob(os.path.join(data_dir, "sub_*")))
train_labels = []
for sub in subjects:
    scan_dirs = sorted(glob.glob(os.path.join(sub, "scan_*")))
    if not scan_dirs: continue
    lbl_path = os.path.join(scan_dirs[0], "segmentation", "gt.nii.gz")
    if os.path.exists(lbl_path):
        train_labels.append(lbl_path)

# 只用训练集生成 (和你之前的逻辑一致)
train_labels = train_labels[:-9] 
print(f"Generating probability map from {len(train_labels)} subjects...")
print(f"Target Shape: {target_shape}")

# 2. 初始化累加器 (H, W, D, 2)
# 通道0: CSF/SAS, 通道1: 侧脑室
sum_map = np.zeros(target_shape + (2,), dtype=np.float32)

# 定义缩放器 (使用最近邻插值，因为是标签)
# 这里的 spatial_size 必须与 sum_map 的前三维一致
resizer = Resize(spatial_size=target_shape, mode='nearest')

# 3. 循环累加
for lbl_path in tqdm(train_labels):
    # 加载数据
    lbl_obj = nib.load(lbl_path)
    lbl_data = lbl_obj.get_fdata() 
    
    # 【关键修复】将数据增加一个通道维度，以便 MONAI 处理
    # (H, W, D) -> (1, H, W, D)
    lbl_tensor = torch.tensor(lbl_data).unsqueeze(0)
    
    # 【关键修复】强制缩放到标准尺寸 (256, 256, 256)
    resized_tensor = resizer(lbl_tensor)
    
    # 移除通道维度，变回 numpy
    # (1, 256, 256, 256) -> (256, 256, 256)
    lbl_data_fixed = resized_tensor.squeeze(0).numpy()
    
    # 通道 0: 累加 CSF (Label 1)
    sum_map[..., 0] += (lbl_data_fixed == 1).astype(np.float32)
    # 通道 1: 累加 侧脑室 (Label 4)
    sum_map[..., 1] += (lbl_data_fixed == 4).astype(np.float32)

# 4. 计算平均值 (概率)
prob_map = sum_map / len(train_labels)

# 5. 保存结果
# 注意：我们使用单位矩阵作为仿射变换，因为这是一个“平均”空间
affine = np.eye(4)
prob_img = nib.Nifti1Image(prob_map, affine)
nib.save(prob_img, save_path)

print("-" * 30)
print(f"Success! Probability map saved to:")
print(save_path)
print(f"Shape: {prob_img.shape}")
print("-" * 30)