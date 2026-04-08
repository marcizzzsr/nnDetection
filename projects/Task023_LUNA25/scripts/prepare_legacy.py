"""
The following script prepares the LUNA25 dataset, along with masks created using MedSAMv2 found at https://huggingface.co/datasets/wanglab/LUNA25-MedSAM2
to match the convention of nnDetection. These masks have a unique value associated to each lesion with no information regarding the lesion label.
The script will create a JSON file for each mask that maps each unique lesion ID to its label class from the CSV.
nnDetection requires masks with unique IDs and a separate JSON file with the class mapping.

According to nnDetection authors, the script will also rename the filenames by substituing "." with "_" to avoid confusing the scripts
"""

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import pandas as pd
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from loguru import logger
from nndet.io import save_json
from nndet.utils.check import env_guard


def copy_file(src, dst):
    """Helper to copy file with directory creation."""
    if not src.exists():
        return False
    shutil.copy2(src, dst)
    return True


def convert_mha_to_nifti(src, dst):
    """Helper to convert mha to nifti."""
    if not src.exists():
        return False
    try:
        img = sitk.ReadImage(str(src))
        sitk.WriteImage(img, str(dst))
        return True
    except Exception as e:
        logger.debug(f"Error converting {src} to {dst}: {e}")
        return False


def process_case(
    uid, image_dir, mask_dir, output_dir, is_train, annotations_group, labels_only=False
):
    """
    Process a single case: copy image and mask, create JSON with instance mappings.
    """
    # Sanitize uid: replace dots with underscores (nnDetection doesn't like dots in filenames)
    uid_sanitized = uid.replace(".", "_")

    # Find image file (handle .mha)
    if not labels_only:
        src_img = image_dir / f"{uid}.mha"
        if not src_img.exists():
            return uid, False, f"Image not found at {src_img} path"

    # Determine destination paths
    if is_train:
        dst_img = output_dir / "imagesTr" / f"{uid_sanitized}_0000.nii.gz"
        dst_lbl = output_dir / "labelsTr" / f"{uid_sanitized}.nii.gz"
        dst_json = output_dir / "labelsTr" / f"{uid_sanitized}.json"
    else:
        dst_img = output_dir / "imagesTs" / f"{uid_sanitized}_0000.nii.gz"
        dst_lbl = output_dir / "labelsTs" / f"{uid_sanitized}.nii.gz"
        dst_json = output_dir / "labelsTs" / f"{uid_sanitized}.json"

    # Handle Mask
    src_mask = mask_dir / f"{uid}.nii.gz"
    if not src_mask.exists():
        src_mask = mask_dir / f"{uid}.nii"
        if not src_mask.exists():
            return uid, False, "Mask not found"

    if not copy_file(src_mask, dst_lbl):
        return uid, False, "Failed to copy mask"

    # Create JSON with instance-to-class mappings
    try:
        # Load the mask to get instance IDs
        nii = nib.load(src_mask)
        data = nii.get_fdata()
        affine = nii.affine

        # Dictionary to map lesion_id to class label
        instance_mapping = {}

        # Inverse affine to convert world coordinates to voxel coordinates
        inv_affine = np.linalg.inv(affine)

        for _, row in annotations_group.iterrows():
            # Get coordinates (annotations are in LPS, need to convert to RAS)
            world_coords = np.array([-row["CoordX"], -row["CoordY"], row["CoordZ"], 1.0])

            # Convert to voxel coords
            voxel_coords = inv_affine @ world_coords
            x, y, z = np.round(voxel_coords[:3]).astype(int)

            # Check bounds
            if (
                (0 <= x < data.shape[0])
                and (0 <= y < data.shape[1])
                and (0 <= z < data.shape[2])
            ):
                # Get the unique ID at this location
                lesion_id = int(data[x, y, z])

                if lesion_id == 0:
                    continue

                # Get the class label from CSV (0: non-lesion, 1: lesion)
                csv_label = int(row["label"])

                # Map the lesion_id to its class label
                instance_mapping[str(lesion_id)] = csv_label

        ids = np.unique(data)
        ids = ids[ids>0]

        if len(ids) != len(instance_mapping):
            logger.warn(f"Found {len(ids)} uniqu ids in {uid} but json instances contain {len(instance_mapping)}. (check coordinate system conversion?)")
            return uid, False, "Instances mismatch"

        # Save the JSON file with instance mapping
        json_data = {"instances": instance_mapping}
        with open(dst_json, "w") as f:
            json.dump(json_data, f, indent=4)

    except Exception as e:
        return uid, False, f"Failed to create JSON: {e}"

    # Convert Image (skip if labels_only)
    if not labels_only:
        if not convert_mha_to_nifti(src_img, dst_img):
            return uid, False, "Failed to convert image"

    return uid, True, None


@env_guard
def main():
    """
    Prepare LUNA25 dataset for nnDetection.
    Uses CT scans with lung nodule annotations.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Prepare LUNA25 dataset for nnDetection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images", type=str, required=True, help="Path to original LUNA25 images"
    )
    parser.add_argument(
        "--masks", type=str, required=True, help="Path to processed masks"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to nnDetection raw data output directory",
    )
    parser.add_argument(
        "-c", "--csv", type=str, required=True, help="Path to the annotations CSV file."
    )
    parser.add_argument(
        "--train-csv", type=str, required=True, help="Path to luna25-train.csv"
    )
    parser.add_argument(
        "--test-csv", type=str, required=True, help="Path to luna25-test.csv"
    )
    parser.add_argument("--jobs", type=int, default=8, help="Number of parallel jobs")
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Only process labels (masks and JSON files), skip image conversion",
    )
    args = parser.parse_args()

    overall_start_time = time.time()

    overall_start_time = time.time()

    task_data_dir = Path(args.output) / "Task023_LUNA25"
    task_data_dir.mkdir(parents=True, exist_ok=True)

    # Setup enhanced loguru logging
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
    logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Setup raw paths
        image_dir = Path(args.images)
        mask_dir = Path(args.masks)

        # Validate input paths
        logger.info("Validating input paths...")
        if not image_dir.exists():
            raise RuntimeError(f"Images directory not found: {image_dir}")
        if not mask_dir.exists():
            raise RuntimeError(f"Masks directory not found: {mask_dir}")
        logger.info("✓ All input paths validated")

        # Setup target paths
        logger.info("Creating output directories...")
        data_target = task_data_dir / "raw_splitted" / "imagesTr"
        data_target.mkdir(parents=True, exist_ok=True)
        label_target = task_data_dir / "raw_splitted" / "labelsTr"
        label_target.mkdir(parents=True, exist_ok=True)
        test_data_target = task_data_dir / "raw_splitted" / "imagesTs"
        test_data_target.mkdir(parents=True, exist_ok=True)
        test_label_target = task_data_dir / "raw_splitted" / "labelsTs"
        test_label_target.mkdir(parents=True, exist_ok=True)
        logger.info("✓ Output directories created")

        # Load CSVs
        logger.info("\n ------- Loading Train/Test Splits from CSV")
        train_df = pd.read_csv(args.train_csv)
        test_df = pd.read_csv(args.test_csv)
        annotations_csv = pd.read_csv(args.csv)

        train_uids = train_df["SeriesInstanceUID"].unique()
        test_uids = test_df["SeriesInstanceUID"].unique()

        logger.info(
            f"Found {len(train_uids)} training cases and {len(test_uids)} test cases"
        )

        training_entries = []
        test_entries = []

        # Phase 1: Process Training Data
        logger.info("\n ------- Processing Training Cases")
        train_total = len(train_uids)
        train_success = 0

        logger.info(f"Processing {train_total} train cases with {args.jobs} workers")
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = []
            for uid in train_uids:
                # Get annotations for this case
                annotations = annotations_csv[
                    annotations_csv["SeriesInstanceUID"] == uid
                ]
                futures.append(
                    executor.submit(
                        process_case,
                        uid,
                        image_dir,
                        mask_dir,
                        task_data_dir / "raw_splitted",
                        True,
                        annotations,
                        args.labels_only,
                    )
                )

            for future in tqdm(futures, total=len(train_uids), desc="Training cases"):
                uid, success, error = future.result()
                if success:
                    uid_sanitized = uid.replace(".", "_")
                    training_entries.append(f"{uid_sanitized}.nii.gz")
                    train_success += 1
                else:
                    logger.warning(f"Skipping train case {uid}: {error}")

        # Phase 2: Process Test Data
        logger.info("\n ------- Processing Test Cases")
        test_total = len(test_uids)
        test_success = 0

        logger.info(f"Processing {test_total} test cases with {args.jobs} workers")
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = []
            for uid in test_uids:
                # Get annotations for this case
                annotations = annotations_csv[
                    annotations_csv["SeriesInstanceUID"] == uid
                ]
                futures.append(
                    executor.submit(
                        process_case,
                        uid,
                        image_dir,
                        mask_dir,
                        task_data_dir / "raw_splitted",
                        False,
                        annotations,
                        args.labels_only,
                    )
                )

            for future in tqdm(futures, total=len(test_uids), desc="Test cases"):
                uid, success, error = future.result()
                if success:
                    uid_sanitized = uid.replace(".", "_")
                    test_entries.append(f"{uid_sanitized}.nii.gz")
                    test_success += 1
                else:
                    logger.warning(f"Skipping test case {uid}: {error}")

        # Save dataset metadata
        logger.info("Creating dataset metadata...")
        dataset_info = {
            "name": "LUNA25",
            "task": "Task023_LUNA25",
            "target_class": None,
            "test_labels": True,
            "labels": {"0": "non-lesion", "1": "lesion"},
            "modalities": {"0": "CT"},
            "dim": 3,
            "info": "LUNA25 dataset for lung nodule detection. Train/test splits from CSV files.",
        }
        save_json(dataset_info, task_data_dir / "dataset.json")
        logger.info("✓ Dataset metadata saved")

        # Create nnDetection format splits
        final_splits = {
            "train": training_entries,
            "test": test_entries,
        }
        save_json(final_splits, task_data_dir / "raw_splitted" / "splits.json")

        # Final summary
        overall_time = time.time() - overall_start_time

        logger.info("\n" + "=" * 60)
        logger.info(" ------- PREPARATION COMPLETED")
        logger.info("=" * 60)
        logger.info(
            f"Training: {train_success}/{train_total} cases ({train_success / train_total * 100:.1f}%)"
        )
        logger.info(
            f"Test: {test_success}/{test_total} cases ({test_success / test_total * 100:.1f}%)"
        )
        logger.info(
            f"Total time: {overall_time:.2f}s ({overall_time / 60:.1f} minutes)"
        )
        logger.info(f"Output: {task_data_dir}")

        if train_success < train_total or test_success < test_total:
            logger.warning("Some cases failed - check logs for details")

    except Exception as e:
        error_time = time.time() - overall_start_time
        logger.error(f"Preparation failed after {error_time:.2f}s: {e}")
        logger.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
