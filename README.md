# Automated Lateral Ventricle Segmentation in Pediatric Hydrocephalus

This repository contains the core codebase for the automated segmentation of lateral ventricles in pediatric hydrocephalus MRI. The framework is built upon a Large SwinUNETR architecture, optimized with boundary-aware loss fine-tuning and Largest Connected Component (LCC) post-processing.

## Model Performance
Evaluated on a clinical pediatric cohort (43 cases):
- **LV Dice (after LCC):** 0.9087
- **LV HD95:** 2.4308 mm

## Repository Structure
- `src/`: Core implementation including network definitions, data preprocessing pipelines, and the optimal training/evaluation scripts.
- `splits/`: Train/validation/test dataset split configurations.
- `*.sh`: SLURM batch scripts for model training and deployment on high-performance computing clusters.

*Note: Clinical NIfTI/DICOM data and heavy pre-trained weights are omitted from this repository to comply with patient privacy regulations and storage limits.*
