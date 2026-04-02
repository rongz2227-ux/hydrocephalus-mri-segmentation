import os
import glob
import argparse
import numpy as np
import nibabel as nib
from scipy import ndimage


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    mask: 3D binary mask, values in {0,1}
    return: 3D binary mask after keeping largest connected component
    """
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask.astype(np.uint8)

    component_sizes = ndimage.sum(mask, labeled, index=range(1, num + 1))
    largest_component = np.argmax(component_sizes) + 1
    largest_mask = (labeled == largest_component).astype(np.uint8)
    return largest_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing *_pred_on_raw.nii.gz")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save *_pred_lv_lcc.nii.gz")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pred_paths = sorted(glob.glob(os.path.join(args.input_dir, "*_pred_on_raw.nii.gz")))
    print(f"Found {len(pred_paths)} LV prediction files.")

    for pred_path in pred_paths:
        nii = nib.load(pred_path)
        mask = nii.get_fdata()
        mask = (mask > 0).astype(np.uint8)

        lcc_mask = keep_largest_connected_component(mask)

        base = os.path.basename(pred_path).replace("_pred_lv.nii.gz", "_pred_lv_lcc.nii.gz")
        out_path = os.path.join(args.output_dir, base)

        out_nii = nib.Nifti1Image(lcc_mask.astype(np.uint8), nii.affine, nii.header)
        nib.save(out_nii, out_path)

        print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()