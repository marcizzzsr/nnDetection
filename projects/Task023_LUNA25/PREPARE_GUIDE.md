# Task023_LUNA25 Preparation Guide

## Overview

Primary script: `scripts/prepare_fused.py`

This standalone script prepares LUNA25 for nnDetection and combines:
- full dataset assembly (`imagesTr/imagesTs`, `labelsTr/labelsTs`, `dataset.json`, `splits.json`)
- robust MedSAM2 mask-to-label mapping diagnostics (coordinate fallback + local search + detailed conversion log)

Legacy scripts are still present for reference:
- `scripts/prepare.py`
- `scripts/prepare_MedSAMv2_masks.py`

## Input Requirements

Expected input layout:

```text
LUNA25/
├── luna25_images/              # CT volumes in .mha
├── medsam2_masks/              # MedSAM2 instance masks in .nii/.nii.gz
├── luna25-annotations.csv      # Lesion coordinates + labels
├── luna25-train.csv            # Train split with SeriesInstanceUID
└── luna25-test.csv             # Test split with SeriesInstanceUID
```

Important:
- The old `nodule_blocks` folder is not used in the current pipeline.
- Masks are read from the MedSAM2 mask directory.

## CSV Columns

Required in annotations CSV:
- `SeriesInstanceUID`
- `CoordX`
- `CoordY`
- `CoordZ`
- `label`

Required in train/test split CSVs:
- `SeriesInstanceUID`

## Usage

### Full pipeline

```bash
python scripts/prepare_fused.py \
  --images /path/to/luna25_images \
  --masks /path/to/medsam2_masks \
  --csv /path/to/luna25-annotations.csv \
  --train-csv /path/to/luna25-train.csv \
  --test-csv /path/to/luna25-test.csv \
  --output /path/to/output_root \
  --jobs 8
```

### Labels only (skip image conversion)

```bash
python scripts/prepare_fused.py \
  --images /path/to/luna25_images \
  --masks /path/to/medsam2_masks \
  --csv /path/to/luna25-annotations.csv \
  --train-csv /path/to/luna25-train.csv \
  --test-csv /path/to/luna25-test.csv \
  --output /path/to/output_root \
  --labels-only
```

### Useful options

- `--local-search-radius`: radius for local voxel neighborhood search (default: `1`)
- `--log-csv`: custom path for conversion CSV log
- `--no-strict-instance-match`: allow partial mappings instead of failing mismatch cases

## Output Structure

```text
Task023_LUNA25/
├── dataset.json
├── prepare.log
├── conversion_log.csv
└── raw_splitted/
    ├── imagesTr/
    ├── labelsTr/
    ├── imagesTs/
    ├── labelsTs/
    └── splits.json
```

Notes:
- Label masks are instance masks (`0` = background, `>0` = lesion instance id)
- JSON files map instance id to class label: `{ "instances": { "1": 1, ... } }`

## What The Fused Script Adds

Compared to the older `prepare.py`, the fused script also logs:
- native coordinate hits
- flipped-XY coordinate hits
- local-search hits
- out-of-bounds / background misses
- matched and missing instance ids per case

These are stored in `conversion_log.csv`.

## Troubleshooting

1. `Mask not found`
- Ensure mask file stem matches `SeriesInstanceUID` (or sanitized version with `.` replaced by `_`).

2. `Instances mismatch`
- Check coordinate convention issues.
- Review `conversion_log.csv` for `missing_instance_ids` and hit counters.
- If needed, run with `--no-strict-instance-match`.

3. Slow processing
- Lower `--jobs` if I/O is saturated.
- Set `--local-search-radius 0` to disable neighborhood search.
