import os, argparse
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation, inv_ornt_aff

def reorient_to_RAS(nii):
    data = nii.get_fdata()
    aff = nii.affine
    cur = io_orientation(aff)
    ras = axcodes2ornt(("R","A","S"))
    trn = ornt_transform(cur, ras)
    data2 = apply_orientation(data, trn)
    aff2 = aff @ inv_ornt_aff(trn, data.shape)
    hdr = nii.header.copy()
    hdr.set_qform(aff2, code=1)
    hdr.set_sform(aff2, code=1)
    return nib.Nifti1Image(data2, aff2, header=hdr)

def make_orthogonal_affine(img_ras, voxel=1.0):
    A = img_ras.affine.copy()
    M = A[:3,:3]
    # 取方向单位向量（正交化）
    # 用 SVD 做最近的正交矩阵
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    A2 = A.copy()
    A2[:3,:3] = R * voxel
    return A2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--voxel", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    img0 = nib.load(args.img)
    gt0  = nib.load(args.gt)
    pr0  = nib.load(args.pred)

    # 1) 先转到 RAS（方向一致）
    img_ras = reorient_to_RAS(img0)

    # 2) 构造“正交 RAS + 1mm”目标网格（去 oblique）
    A_ortho = make_orthogonal_affine(img_ras, voxel=args.voxel)

    # 保持 FOV 大致一致：按体素尺寸比例估算 shape
    # 当前体素大小（列向量范数）
    vx = np.sqrt((img_ras.affine[:3,:3]**2).sum(0))
    old_shape = np.array(img_ras.shape[:3], dtype=float)
    new_shape = np.ceil(old_shape * (vx / args.voxel)).astype(int)

    target = (tuple(new_shape.tolist()), A_ortho)

    # 3) 把三者都 resample 到同一个正交网格
    img_out = resample_from_to(img_ras, target, order=1)      # bilinear
    gt_out  = resample_from_to(gt0, target, order=0)          # nearest
    pr_out  = resample_from_to(pr0, target, order=0)          # nearest

    # 4) 强制 qform/sform 一致（避免 ITK-SNAP header mismatch）
    for nii in (img_out, gt_out, pr_out):
        nii.header.set_qform(img_out.affine, code=1)
        nii.header.set_sform(img_out.affine, code=1)

    nib.save(img_out, os.path.join(args.out_dir, "img_ortho_RAS1mm.nii.gz"))
    nib.save(gt_out,  os.path.join(args.out_dir, "gt_ortho_RAS1mm.nii.gz"))
    nib.save(pr_out,  os.path.join(args.out_dir, "pred_ortho_RAS1mm.nii.gz"))

    print("[DONE] saved to", args.out_dir)
    print("shape:", img_out.shape)

if __name__ == "__main__":
    main()
