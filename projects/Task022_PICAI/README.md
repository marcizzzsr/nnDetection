# Task022_PICAI - PI-CAI Dataset for nnDetection

This task prepares the PI-CAI (Prostate Imaging: Cancer AI) dataset for use with nnDetection framework, enabling both detection and instance segmentation of clinically significant prostate cancer lesions.

## Dataset Overview

- **Task**: Instance segmentation and detection of clinically significant prostate cancer (csPCa) lesions
- **Modalities**: T2-weighted (T2w), Apparent Diffusion Coefficient (ADC), High b-value DWI (HBV)  
- **Data**: Multi-parametric MRI scans with expert-annotated lesion masks
- **Target**: csPCa lesions (binary classification: background vs lesion)

## Dataset Structure

The raw dataset should be organized as follows under `$det_data/Task022_PICAI/raw/`:

```
picai_extracted_images/
├── 10000/
│   ├── 10000_1000000_t2w.mha
│   ├── 10000_1000000_adc.mha
│   ├── 10000_1000000_hbv.mha
│   ├── 10000_1001000_t2w.mha
│   └── ...
├── 10001/
│   └── ...
└── ...

picai_labels/
└── csPCa_lesion_delineations/
    ├── 10000_1000000.nii.gz
    ├── 10000_1001000.nii.gz
    └── ...
```

## Custom Splits

This preparation script is able to handle a custom train-test split file that can be passed at runtime with `--splits_file` argument.

The splits file should contain:
```json
{
  "test": ["case_id_1", "case_id_2", ...],
  "train": ["case_id_3", "case_id_4", ...]  // Optional
}
```

If `train` is not specified, all cases not in `test` will be used for training.

## Preparation Process

The script performs the following steps:

1. **Case Discovery**: Scans the dataset to find all valid cases with complete modalities
2. **Image Processing**: 
   - Loads all three modalities (T2w, ADC, HBV) per case
   - **Resamples ADC and HBV to match T2w spatial resolution**
   - **No center cropping is performed**
   - Saves images with modality indexing (0000=T2w, 0001=ADC, 0002=HBV)
3. **Label Processing**:
   - Loads lesion masks and resamples to match image space
   - Performs connected component analysis to create instance segmentation
   - Generates instance metadata (all lesions are class 0: csPCa) https://github.com/MIC-DKFZ/nnDetection/issues/276
   - Creates empty labels for test cases without annotations
4. **Split Application**: Applies custom train/test splits
5. **Output Generation**: Creates nnDetection-compatible directory structure

## Usage

### Prerequisites

Ensure the nnDetection environment is properly configured:
```bash
export det_data="/path/to/nndetection/data"
export det_models="/path/to/nndetection/models"  
export det_num_threads=6
export OMP_NUM_THREADS=1
```

### Run Preparation

```bash
cd /path/to/nnDetection/projects/Task022_PICAI/scripts
python prepare.py \
      --source-image-dir <path-to-raw-images> \
      --source-labels-dir <path-to-raw-labels> \
      --output-dir <path-to-output-directory> \
      --splits_file <path-to-custom-splits> \
      -npp <number-of-processes> # defaults to 4
```

### Expected Output

The script creates the following structure:

```
$det_data/Task022_PICAI/
├── dataset.json                    # Dataset metadata
├── splits.json                     # Case splits for reference
├── prepare.log                     # Detailed preparation log
└── raw_splitted/
    ├── splits.json                 # nnDetection format splits
    ├── imagesTr/                   # Training images
    │   ├── case_id_0000.nii.gz     # T2w modality
    │   ├── case_id_0001.nii.gz     # ADC modality 
    │   ├── case_id_0002.nii.gz     # HBV modality
    │   └── ...
    ├── labelsTr/                   # All labels (train + test)
    │   ├── case_id.nii.gz          # Instance segmentation mask
    │   ├── case_id.json            # Instance metadata
    │   └── ...
    └── imagesTs/                   # Test images
        ├── case_id_0000.nii.gz
        └── ...
```

## Dataset Metadata

- **Name**: PI-CAI
- **Dimensions**: 3D
- **Classes**: 
  - 0: csPCa lesion
- **Modalities**:
  - 000: T2w (T2-weighted)
  - 001: ADC (Apparent Diffusion Coefficient)
  - 002: HBV (High b-value DWI)

## Key Features

- **Multi-modal Input**: Combines T2w anatomical and DWI functional information
- **Instance Segmentation**: Separates individual lesions for detection metrics
- **Custom Splits**: Respects predefined train/test partitions
- **Robust Processing**: Handles missing modalities and empty labels gracefully
- **Connected Components**: Automatically separates touching lesions
- **Space Alignment**: Ensures all modalities are in the same spatial reference

## Troubleshooting

### Common Issues

1. **Missing Modalities**: Check that all cases have T2w, ADC, and HBV files
2. **Path Errors**: Verify `det_data` environment variable and dataset structure
3. **Split Mismatches**: Ensure case IDs in splits file match dataset filenames
4. **Memory Issues**: Large images may require more RAM; consider processing subsets

### Logs

Check `prepare.log` for detailed processing information and error messages.

## Training

After preparation, train the model using nnDetection:

```bash
nndet train 022 3d_fullres 0 --debug
```

## Inference

Run inference on test cases:

```bash
nndet predict 022 3d_fullres 0 -chk model_best -o /path/to/predictions
```

## References

- [PI-CAI Challenge](https://pi-cai.grand-challenge.org/)
- [nnDetection Framework](https://github.com/MIC-DKFZ/nnDetection)
- [Dataset Paper](https://arxiv.org/abs/2103.03404)
