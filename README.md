# Pediatric Hydrocephalus Lateral Ventricle Segmentation

Automated segmentation of the lateral ventricle in pediatric hydrocephalus MRI using Large SwinUNETR, with boundary-aware fine-tuning and quantitative volumetric analysis.

## Background

Accurate segmentation of the lateral ventricle is critical for the clinical assessment of hydrocephalus severity and treatment monitoring. This project develops and evaluates an automated segmentation pipeline on real clinical MRI data collected from a hospital, targeting pediatric patients (age 0.2–17.7 years).

## Dataset

- 43 pediatric hydrocephalus cases, T1-weighted MRI (eT1W FFE SVR)
- Manually corrected segmentation labels by a neurologist (6 classes: background, CSF, gray matter, white matter, lateral ventricle, choroid plexus)
- Split: 31 train / 8 val / 9 test (no data leakage verified)
- Clinical metadata available: age, gender, etiology, hydrocephalus classification, severity

> Raw data is not included in this repository due to patient privacy.

## Method

### Stage 1: Baseline Segmentation (`train_large.py`)

Large SwinUNETR trained from scratch on the full multi-class segmentation task.

| Setting | Value |
|---|---|
| Model | SwinUNETR (feature\_size=48) |
| Input ROI | 128 × 128 × 128 |
| Optimizer | AdamW (lr=1e-4, weight\_decay=1e-5) |
| Loss | DiceCELoss |
| Epochs | 600 |
| Sliding window overlap | 0.25 (constant weighting) |

### Stage 2: Boundary-Aware Fine-tuning (`train_boundary_finetune.py`)

Fine-tuning on the pre-trained model with an additional boundary-weighted loss focused on the lateral ventricle.

| Setting | Value |
|---|---|
| Optimizer | AdamW (lr=1e-5, weight\_decay=1e-5) |
| Loss | DiceCELoss + α × BoundaryWeightedDiceLoss |
| Alpha (α) | 0.5 |
| Epochs | 200 |

`BoundaryWeightedDiceLoss` is implemented in `custom_losses.py`. It extracts the boundary of the ground truth mask using morphological erosion and applies higher loss weight to boundary voxels.

### Post-processing

Largest Connected Component (LCC) filtering applied to the lateral ventricle prediction to remove spurious isolated voxels and improve boundary quality.

## Results

### Model Comparison (LV = Lateral Ventricle)

| Model | Mean Dice | LV Dice | LV HD95 (mm) |
|---|---|---|---|
| Baseline UNet | 0.584 | 0.826 | 5.76 |
| Optimized UNet | 0.672 | 0.857 | 4.08 |
| SwinUNETR | 0.744 | 0.876 | 3.74 |
| Large SwinUNETR | 0.777 | 0.896 | 2.91 |

### Sliding Window Ablation (Large SwinUNETR)

| Overlap | LV Dice | LV HD95 (mm) |
|---|---|---|
| 0.25 | 0.8986 | 2.765 |
| 0.50 | 0.8961 | 2.906 |
| 0.75 | 0.8975 | 2.852 |

### Post-processing & Fine-tuning

| Setting | LV Dice | LV HD95 (mm) |
|---|---|---|
| Baseline (overlap=0.25, constant) | 0.8988 | 2.319 |
| + LCC post-processing | 0.8988 | 2.319 |
| + Boundary Loss (α=0.3) + LCC | 0.9087 | 2.431 |

### Volume Estimation vs. Ground Truth

Predicted lateral ventricle volumes show high agreement with manually annotated GT volumes (R² = 0.987, MAE = 12.4 cm³), validating the pipeline for automated volumetric quantification.

## Quantitative Analysis

Lateral ventricle volume and Ventricle-to-Brain Ratio (VBR) were extracted for all 43 cases and analysed against clinical metadata (severity, etiology). Key findings:

- VBR correlates with clinical severity: mean VBR 0.038 (improving) vs. 0.089 (stable) vs. 0.107 (worsening)
- Congenital and idiopathic hydrocephalus show larger ventricle volumes than tumour-related or post-traumatic cases
- Lateral ventricle volume is not significantly correlated with age within this cohort (R² ≈ 0.01)

## Repository Structure

```
src/
├── train_large.py                # Stage 1: baseline training
├── train_boundary_finetune.py    # Stage 2: boundary-aware fine-tuning
├── custom_losses.py              # BoundaryWeightedDiceLoss implementation
├── eval_compare.py               # Model evaluation (Dice, HD95)
├── eval_lv_lcc.py                # Evaluation with LCC post-processing
├── postprocess_lv_lcc.py         # LCC post-processing
├── export_pred_test.py           # Run inference and save predictions
├── extract_lv_features.py        # Extract LV volume features
├── check_data_leakage.py         # Verify train/val/test split integrity
└── dataset_check.py              # Dataset validation

splits/
├── train_subjects.txt
├── val_subjects.txt
└── test_subjects.txt
```

## Requirements

```
monai
torch
nibabel
scipy
numpy
pandas
```

Install:

```bash
conda create -n hydro_seg python=3.8
conda activate hydro_seg
pip install monai torch nibabel scipy numpy pandas
```

## Usage

### Training

```bash
# Stage 1: baseline
python src/train_large.py

# Stage 2: fine-tuning
python src/train_boundary_finetune.py
```

Or submit to SLURM:

```bash
sbatch run_large.sh
sbatch run_boundary_finetune.sh
```

### Evaluation

```bash
python src/eval_compare.py \
  --data_dir data/ \
  --split_file splits/test_subjects.txt \
  --model_type swin \
  --model_path experiments_large/best_metric_model_large.pth \
  --roi 128 \
  --overlap 0.25 \
  --sw_mode constant \
  --output_csv results/eval_results.csv
```

### Inference on new data

```bash
python src/export_pred_test.py \
  --model_path experiments_large/best_metric_model_large.pth \
  --split_file splits/test_subjects.txt \
  --out_dir preds_output \
  --roi 128 128 128 \
  --overlap 0.25
```

### Volume extraction

```bash
python src/extract_lv_features.py
```

## Acknowledgements

This work uses [MONAI](https://monai.io/) and the [SwinUNETR](https://arxiv.org/abs/2201.01266) architecture. 
