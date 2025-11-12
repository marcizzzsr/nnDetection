# Task023_LUNA25 Preparation Guide

## Overview

Preparation script for the LUNA25 dataset for pulmonary nodule detection using nnDetection framework.

## Dataset Structure

### Input Requirements

The script expects the following input structure:

```
LUNA25/
├── luna25_images/              # Raw CT images in .mha format
├── luna25_nodule_blocks/
│   ├── image/                  # Nodule masks in .npy format
│   └── metadata/               # Nodule metadata in .npy format
└── luna25-annotations.csv      # Mapping between images and nodules
```

### CSV Format

The annotations CSV must contain the following columns:
- `SeriesInstanceUID`: Unique identifier for each CT scan
- `NoduleID`: Unique identifier for each nodule
- `label`: Class label for the nodule (e.g., 0 or 1)

### Output Structure

The script generates the following nnDetection-compatible structure:

```
Task023_LUNA25/
├── dataset.json                # Dataset metadata
├── prepare.log                 # Detailed processing log
├── raw_splitted/
│   ├── imagesTr/              # Converted images (_0000.nii.gz)
│   └── labelsTr/              # Instance masks (.nii.gz) and labels (.json)
└── preprocessed/
    ├── splits_final.pkl       # Train/validation splits (pickle)
    └── splits_final.json      # Train/validation splits (JSON)
```

## Usage

### Basic Command

```bash
python prepare_try.py \
  -i /path/to/luna25_images \
  -m /path/to/luna25_nodule_blocks/image \
  -d /path/to/luna25_nodule_blocks/metadata \
  -a /path/to/luna25-annotations.csv \
  -o /path/to/output
```

### Arguments

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--source-image-dir` | `-i` | Yes | Directory containing raw .mha images |
| `--source-masks-dir` | `-m` | Yes | Directory containing nodule mask blocks (.npy) |
| `--source-metadata-dir` | `-d` | Yes | Directory containing nodule metadata (.npy) |
| `--annotations-csv` | `-a` | Yes | Path to annotations CSV file |
| `--output-dir` | `-o` | Yes | Output directory path |
| `--num_processes` | | No | Number of parallel processes (default: 4) |

### Example

```bash
python prepare_try.py \
  -i /opt/data/LUNA25/luna25_images \
  -m /opt/data/LUNA25/luna25_nodule_blocks/image \
  -d /opt/data/LUNA25/luna25_nodule_blocks/metadata \
  -a /opt/data/luna25-annotations.csv \
  -o /opt/data \
  --num_processes 8
```

## Processing Steps

1. **Input Validation**: Verifies all input directories and CSV file exist and are properly formatted
2. **CSV Validation**: Checks for required columns and reports data statistics
3. **Image Conversion**: Converts .mha images to .nii.gz format with _0000 suffix
4. **Mask Creation**: 
   - Loads individual nodule masks and metadata
   - Combines multiple nodules per image into instance segmentation masks
   - Creates JSON files mapping instance IDs to class labels
5. **Split Generation**: Creates 5-fold cross-validation splits based on unique series

## Output Files

### Images
- Format: `{SeriesInstanceUID}_0000.nii.gz`
- Location: `raw_splitted/imagesTr/`

### Labels
- Mask: `{SeriesInstanceUID}.nii.gz` (instance segmentation with unique integer per nodule)
- JSON: `{SeriesInstanceUID}.json` (maps instance ID to class label)
- Location: `raw_splitted/labelsTr/`

### Splits
- 5-fold cross-validation splits
- Each fold contains `train` and `val` case lists
- Available in both pickle and JSON formats

## Notes

- Instance IDs in masks are sequential integers starting from 1
- Background is represented by 0 in instance masks
- All nodules from the same image are combined into a single instance mask
- Metadata (origin, spacing, direction) is preserved from the original nodule blocks
- Processing is parallelized using multiprocessing for efficiency

## Troubleshooting

### CSV Validation Errors
If you encounter CSV validation errors, ensure your CSV contains the required columns: `SeriesInstanceUID`, `NoduleID`, and `label`.

### Missing Files
If masks or metadata files are missing for certain nodules, the script will log warnings and continue processing remaining nodules.

### Memory Issues
If you encounter memory issues with multiprocessing, reduce the `--num_processes` parameter.
