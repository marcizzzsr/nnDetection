import argparse
import sys
import traceback
from itertools import repeat
from multiprocessing.pool import Pool
from pathlib import Path

import pandas as pd
import numpy as np
import SimpleITK as sitk
from loguru import logger

from nndet.io.load import save_pickle, save_json
from nndet.utils.check import env_guard


def convert_mha_to_nifti(source_dir: Path, target_dir: Path, num_processes: int):
    """Convert .mha images to .nii.gz format with _0000 suffix"""
    mha_files = list(source_dir.glob('*.mha'))
    
    with Pool(processes=num_processes) as p:
        p.starmap(_convert_single_image, zip(mha_files, repeat(target_dir)))


def _convert_single_image(mha_file: Path, target_dir: Path):
    """Convert single .mha file to .nii.gz"""
    try:
        logger.info(f"Converting {mha_file.name}")
        image = sitk.ReadImage(str(mha_file))
        
        # Create output filename with _0000 suffix (modality 0)
        output_name = f"{mha_file.stem}_0000.nii.gz"
        output_path = target_dir / output_name
        
        sitk.WriteImage(image, str(output_path))
    except Exception as e:
        logger.error(f"Failed to convert {mha_file.name}: {e}\n{traceback.format_exc()}")


def create_masks_from_blocks(
    csv_path: Path,
    mask_dir: Path,
    metadata_dir: Path,
    target_dir: Path,
    num_processes: int
):
    """Create instance segmentation masks from nodule blocks"""
    df = pd.read_csv(csv_path)
    
    # Group by SeriesInstanceUID to get all nodules per image
    grouped = df.groupby('SeriesInstanceUID')
    
    tasks = []
    for series_uid, group_df in grouped:
        # Get all nodule info for this image
        nodule_info = []
        for _, row in group_df.iterrows():
            nodule_id = row['NoduleID']
            label = row['label']
            nodule_info.append({
                'nodule_id': nodule_id,
                'label': label,
                'series_uid': series_uid
            })
        
        tasks.append((series_uid, nodule_info, mask_dir, metadata_dir, target_dir))
    
    with Pool(processes=num_processes) as p:
        p.starmap(_create_mask_for_case, tasks)


def _create_mask_for_case(
    series_uid: str,
    nodule_info: list,
    mask_dir: Path,
    metadata_dir: Path,
    target_dir: Path
):
    """Create combined instance mask for a single case"""
    try:
        logger.info(f"Processing mask for {series_uid}")
        
        # Load all nodule masks and metadata
        combined_mask = None
        instance_labels = {}
        instance_id = 1
        
        for info in nodule_info:
            nodule_id = info['nodule_id']
            label = info['label']
            
            # Load mask and metadata
            mask_file = mask_dir / f"{nodule_id}.npy"
            metadata_file = metadata_dir / f"{nodule_id}.npy"
            
            if not mask_file.exists() or not metadata_file.exists():
                logger.warning(f"Missing files for nodule {nodule_id}")
                continue
            
            mask_array = np.load(mask_file)
            metadata = np.load(metadata_file, allow_pickle=True).item()
            
            # Convert to SimpleITK image
            mask_itk = sitk.GetImageFromArray(mask_array)
            mask_itk.SetOrigin(metadata['origin'])
            mask_itk.SetSpacing(metadata['spacing'])
            if 'direction' in metadata:
                mask_itk.SetDirection(metadata['direction'])
            
            # If first mask, initialize combined_mask
            if combined_mask is None:
                combined_mask = sitk.GetArrayFromImage(mask_itk)
                combined_mask = np.zeros_like(combined_mask, dtype=np.uint16)
                reference_metadata = metadata
            
            # Add this nodule to combined mask with unique instance ID
            mask_array = sitk.GetArrayFromImage(mask_itk)
            combined_mask[mask_array > 0] = instance_id
            
            # Store the class label for this instance
            instance_labels[str(instance_id)] = int(label)
            instance_id += 1
        
        if combined_mask is None:
            logger.warning(f"No valid masks found for {series_uid}")
            return
        
        # Convert back to SimpleITK and save
        combined_mask_itk = sitk.GetImageFromArray(combined_mask)
        combined_mask_itk.SetOrigin(reference_metadata['origin'])
        combined_mask_itk.SetSpacing(reference_metadata['spacing'])
        if 'direction' in reference_metadata:
            combined_mask_itk.SetDirection(reference_metadata['direction'])
        
        # Save mask
        output_name = f"{series_uid.replace('.', '_')}.nii.gz"
        sitk.WriteImage(combined_mask_itk, str(target_dir / output_name))
        
        # Save instance labels JSON
        json_name = f"{series_uid.replace('.', '_')}.json"
        save_json({"instances": instance_labels}, target_dir / json_name)
        
    except Exception as e:
        logger.error(f"Failed to create mask for {series_uid}: {e}\n{traceback.format_exc()}")

#TODO
# Need to implement this to maintains same class distribution over splits (stratified k-folds)
def create_splits_from_csv(csv_path: Path, target_dir: Path):
    """Create train/val splits based on CSV data"""
    df = pd.read_csv(csv_path)
    
    # Get unique series UIDs
    all_series = df['SeriesInstanceUID'].unique()
    all_series = [s.replace('.', '_') for s in all_series]
    
    # Create 5-fold cross-validation splits
    splits = []
    n_folds = 5
    fold_size = len(all_series) // n_folds
    
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else len(all_series)
        
        val_ids = all_series[val_start:val_end]
        train_ids = [s for s in all_series if s not in val_ids]
        
        splits.append({
            "train": train_ids,
            "val": val_ids
        })
    
    save_pickle(splits, target_dir / "splits_final.pkl")
    save_json(splits, target_dir / "splits_final.json")


@env_guard
def main():
    parser = argparse.ArgumentParser(
        description="Prepare LUNA25 dataset for nnDetection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--num_processes', 
        type=int, 
        default=4, 
        required=False,
        help="Number of processes to use for preparation."
    )
    parser.add_argument(
        "--source-image-dir",
        "-i",
        type=str,
        required=True,
        help="Directory containing raw .mha images"
    )
    parser.add_argument(
        "--source-masks-dir",
        "-m",
        type=str,
        required=True,
        help="Directory containing nodule mask blocks (.npy files)"
    )
    parser.add_argument(
        "--source-metadata-dir",
        "-d",
        type=str,
        required=True,
        help="Directory containing nodule metadata (.npy files)"
    )
    parser.add_argument(
        "--annotations-csv",
        "-a",
        type=str,
        required=True,
        help="Path to annotations CSV file mapping images to nodules"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="Path to output directory (will contain Task023_LUNA25 structure)"
    )

    parser.add_argument(
        "--create-splits",
        action='store_true',
        help="Create train-val splits based on csv file"
    )
    
    args = parser.parse_args()
    num_processes = args.num_processes

    task_data_dir = Path(args.output_dir) / "Task023_LUNA25"
    task_data_dir.mkdir(parents=True, exist_ok=True)

    # Setup paths from arguments
    image_dir = Path(args.source_image_dir)
    mask_dir = Path(args.source_masks_dir)
    metadata_dir = Path(args.source_metadata_dir)
    csv_path = Path(args.annotations_csv)
    
    # Validate input paths
    logger.info("Validating input paths...")
    if not image_dir.is_dir():
        raise ValueError(f"Image directory does not exist: {image_dir}")
    if not mask_dir.is_dir():
        raise ValueError(f"Mask directory does not exist: {mask_dir}")
    if not metadata_dir.is_dir():
        raise ValueError(f"Metadata directory does not exist: {metadata_dir}")
    if not csv_path.is_file():
        raise ValueError(f"Expected {csv_path} to exist")

    # Create target directories
    logger.info("Creating output directories...")
    target_data_dir = task_data_dir / "raw_splitted" / "imagesTr"
    target_data_dir.mkdir(exist_ok=True, parents=True)
    target_label_dir = task_data_dir / "raw_splitted" / "labelsTr"
    target_label_dir.mkdir(exist_ok=True, parents=True)
    target_preprocessed_dir = task_data_dir / "preprocessed"
    target_preprocessed_dir.mkdir(exist_ok=True)
    logger.info("✓ Output directories created")

    # Setup logging
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")
    logger.add(
        task_data_dir / "prepare.log",
        format="{time} | {level} | {message}",
        level="DEBUG",
    )

    logger.info("=" * 60)
    logger.info("LUNA25 DATASET PREPARATION FOR NNDETECTION")
    logger.info("=" * 60)
    logger.info("=" * 60)

    # Create dataset metadata
    logger.info("Creating dataset metadata...")
    meta = {
        "name": "LUNA25",
        "task": "Task023_LUNA25",
        "target_class": None,
        "test_labels": False,
        "labels": {
            "0": "nodule",
        },
        "modalities": {
            "0": "CT",
        },
        "dim": 3,
    }
    save_json(meta, task_data_dir / "dataset.json")
    logger.info("✓ Dataset metadata saved")

    # Convert images
    logger.info("\n ------- Converting Images")
    convert_mha_to_nifti(image_dir, target_data_dir, num_processes=num_processes)

    # Create masks from nodule blocks
    logger.info("\n ------- Creating Instance Masks")
    create_masks_from_blocks(
        csv_path,
        mask_dir,
        metadata_dir,
        target_label_dir,
        num_processes=num_processes
    )

    # Generate splits
    if args.create_splits:
        logger.info("\n ------- Generating Splits")
        create_splits_from_csv(csv_path, target_preprocessed_dir)
        
    logger.info("\n" + "=" * 60)
    logger.info("PREPARATION COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Output directory: {task_data_dir}")


if __name__ == '__main__':
    main()