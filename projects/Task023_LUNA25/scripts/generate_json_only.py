"""
Generate JSON files for LUNA25 masks with instance-to-class mappings.
This script only creates JSON files and skips all image/mask copying operations.
It expects masks to already exist in the source directory.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm

import SimpleITK as sitk
import pandas as pd
import nibabel as nib
import numpy as np
from loguru import logger
from nndet.utils.check import env_guard


def process_case_json_only(uid, mask_dir, annotations_group, skip_validation=False):
    """
    Process a single case: create JSON with instance mappings for existing mask.
    Returns: (uid, success, error_message, mismatch_info)
    mismatch_info is None or (unique_ids_count, instance_mapping_count)
    """
    # Sanitize uid: replace dots with underscores
    uid_sanitized = uid.replace(".", "_")

    # Find mask file
    mask_path = mask_dir / f"{uid_sanitized}.nii.gz"
    if not mask_path.exists():
        mask_path = mask_dir / f"{uid_sanitized}.nii"
        if not mask_path.exists():
            return uid, False, f"Mask not found at {mask_path}", None

    # JSON will be saved next to the mask
    json_path = mask_dir / f"{uid_sanitized}.json"

    # Create JSON with instance-to-class mappings
    try:
        # Use memory mapping to avoid loading entire file - much faster!
        nii = nib.load(mask_path, mmap=True)

        # Get data without loading to memory yet
        data = nii.dataobj
        affine = nii.affine

        # Dictionary to map lesion_id to class label
        instance_mapping = {}

        # Compute inverse affine once
        inv_affine = np.linalg.inv(affine)

        # Pre-allocate coordinate array for vectorization
        n_annotations = len(annotations_group)
        world_coords_batch = np.ones((n_annotations, 4))

        # Vectorize coordinate transformation
        for idx, (_, row) in enumerate(annotations_group.iterrows()):
            world_coords_batch[idx, 0] = -row["CoordX"]
            world_coords_batch[idx, 1] = -row["CoordY"]
            world_coords_batch[idx, 2] = row["CoordZ"]

        # Batch transform all coordinates at once
        voxel_coords_batch = (inv_affine @ world_coords_batch.T).T[:, :3]
        voxel_coords_batch = np.round(voxel_coords_batch).astype(int)

        # Get shape for bounds checking
        shape = data.shape

        # Process each annotation with vectorized coordinates
        for idx, (_, row) in enumerate(annotations_group.iterrows()):
            x, y, z = voxel_coords_batch[idx]

            # Check bounds
            if not (0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]):
                continue

            # Only read single voxel - much faster with mmap!
            lesion_id = int(data[x, y, z])

            if lesion_id == 0:
                continue

            # Get the class label from CSV
            csv_label = int(row["label"])

            # Map the lesion_id to its class label
            instance_mapping[str(lesion_id)] = csv_label

        # Only load unique IDs if needed for validation
        mismatch_info = None
        if not skip_validation:
            ids = np.unique(np.array(data))
            ids = ids[ids > 0]

            if len(ids) != len(instance_mapping):
                mismatch_info = (len(ids), len(instance_mapping))
                logger.warning(
                    f"Found {len(ids)} unique ids in {uid} but json instances contain "
                    f"{len(instance_mapping)}. (check coordinate system conversion?)"
                )
                # Don't fail, just warn
                # return uid, False, "Instances mismatch", mismatch_info

        # Save the JSON file with instance mapping
        json_data = {"instances": instance_mapping}
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)

    except Exception as e:
        return uid, False, f"Failed to create JSON: {e}", None

    return uid, True, None, mismatch_info


def process_batch(args_tuple):
    """Process a batch of cases - for ProcessPoolExecutor"""
    uids, mask_dir, annotations_csv = args_tuple
    results = []
    for uid in uids:
        annotations = annotations_csv[annotations_csv["SeriesInstanceUID"] == uid]
        result = process_case_json_only(uid, Path(mask_dir), annotations)
        results.append(result)
    return results


@env_guard
def main():
    """
    Generate JSON files for LUNA25 masks with instance-to-class mappings.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate JSON files for LUNA25 masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--masks", type=str, required=True, help="Path to directory containing masks"
    )
    parser.add_argument(
        "-c", "--csv", type=str, required=True, help="Path to the annotations CSV file"
    )
    parser.add_argument(
        "--train-csv", type=str, required=True, help="Path to luna25-train.csv"
    )
    parser.add_argument(
        "--test-csv", type=str, required=True, help="Path to luna25-test.csv"
    )
    parser.add_argument("--jobs", type=int, default=8, help="Number of parallel jobs")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "both"],
        default="both",
        help="Which split to process",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip unique ID validation for speed",
    )
    parser.add_argument(
        "--use-processes",
        action="store_true",
        help="Use ProcessPoolExecutor instead of ThreadPoolExecutor",
    )
    args = parser.parse_args()

    overall_start_time = time.time()

    mask_dir = Path(args.masks)

    # Setup enhanced loguru logging
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")
    logger.add(
        mask_dir / "generate_json.log",
        format="{time} | {level} | {message}",
        level="DEBUG",
    )

    logger.info("=" * 60)
    logger.info("LUNA25 JSON GENERATION")
    logger.info("=" * 60)
    logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Validate input paths
        logger.info("Validating input paths...")
        if not mask_dir.exists():
            raise RuntimeError(f"Masks directory not found: {mask_dir}")
        logger.info("✓ All input paths validated")

        # Load CSVs
        logger.info("\nLoading Train/Test Splits from CSV")
        train_df = pd.read_csv(args.train_csv)
        test_df = pd.read_csv(args.test_csv)
        annotations_csv = pd.read_csv(args.csv)

        train_uids = train_df["SeriesInstanceUID"].unique()
        test_uids = test_df["SeriesInstanceUID"].unique()

        logger.info(
            f"Found {len(train_uids)} training cases and {len(test_uids)} test cases"
        )

        # Choose executor type
        ExecutorClass = (
            ProcessPoolExecutor if args.use_processes else ThreadPoolExecutor
        )

        # Process based on split argument
        total_success = 0
        total_cases = 0
        problematic_cases = []  # Track cases with mismatches

        if args.split in ["train", "both"]:
            logger.info("\n ------- Processing Training Cases")
            train_total = len(train_uids)
            train_success = 0
            total_cases += train_total

            logger.info(
                f"Processing {train_total} train cases with {args.jobs} workers"
            )
            with ExecutorClass(max_workers=args.jobs) as executor:
                futures = []
                for uid in train_uids:
                    annotations = annotations_csv[
                        annotations_csv["SeriesInstanceUID"] == uid
                    ]
                    futures.append(
                        executor.submit(
                            process_case_json_only,
                            uid,
                            mask_dir,
                            annotations,
                            args.skip_validation,
                        )
                    )

                for future in tqdm(
                    futures, total=len(train_uids), desc="Training cases"
                ):
                    uid, success, error, mismatch_info = future.result()
                    if success:
                        train_success += 1
                        if mismatch_info is not None:
                            unique_count, mapping_count = mismatch_info
                            problematic_cases.append(
                                (uid, "train", unique_count, mapping_count)
                            )
                    else:
                        logger.warning(f"Failed to process train case {uid}: {error}")

            total_success += train_success
            logger.info(
                f"Training: {train_success}/{train_total} cases "
                f"({train_success / train_total * 100:.1f}%)"
            )

        if args.split in ["test", "both"]:
            logger.info("\n ------- Processing Test Cases")
            test_total = len(test_uids)
            test_success = 0
            total_cases += test_total

            logger.info(f"Processing {test_total} test cases with {args.jobs} workers")
            with ExecutorClass(max_workers=args.jobs) as executor:
                futures = []
                for uid in test_uids:
                    annotations = annotations_csv[
                        annotations_csv["SeriesInstanceUID"] == uid
                    ]
                    futures.append(
                        executor.submit(
                            process_case_json_only,
                            uid,
                            mask_dir,
                            annotations,
                            args.skip_validation,
                        )
                    )

                for future in tqdm(futures, total=len(test_uids), desc="Test cases"):
                    uid, success, error, mismatch_info = future.result()
                    if success:
                        test_success += 1
                        if mismatch_info is not None:
                            unique_count, mapping_count = mismatch_info
                            problematic_cases.append(
                                (uid, "test", unique_count, mapping_count)
                            )
                    else:
                        logger.warning(f"Failed to process test case {uid}: {error}")

            total_success += test_success
            logger.info(
                f"Test: {test_success}/{test_total} cases "
                f"({test_success / test_total * 100:.1f}%)"
            )

        # Write problematic cases to file
        if problematic_cases:
            problematic_log = mask_dir / "problematic_cases.txt"
            with open(problematic_log, "w") as f:
                f.write("Problematic Cases with Instance Mismatches\n")
                f.write("=" * 80 + "\n")
                f.write(f"Total problematic cases: {len(problematic_cases)}\n\n")
                f.write(
                    f"{'Case UID':<50} {'Split':<8} {'Unique IDs':<12} {'Mapped IDs'}\n"
                )
                f.write("-" * 80 + "\n")
                for uid, split, unique_count, mapping_count in problematic_cases:
                    f.write(
                        f"{uid:<50} {split:<8} {unique_count:<12} {mapping_count}\n"
                    )
            logger.info(
                f"\nWrote {len(problematic_cases)} problematic cases to {problematic_log}"
            )

        # Final summary
        overall_time = time.time() - overall_start_time

        logger.info("\n" + "=" * 60)
        logger.info(" ------- JSON GENERATION COMPLETED")
        logger.info("=" * 60)
        logger.info(
            f"Total: {total_success}/{total_cases} cases "
            f"({total_success / total_cases * 100:.1f}%)"
        )
        logger.info(
            f"Total time: {overall_time:.2f}s ({overall_time / 60:.1f} minutes)"
        )
        logger.info(f"Output directory: {mask_dir}")

        if total_success < total_cases:
            logger.warning("Some cases failed - check logs for details")

    except Exception as e:
        error_time = time.time() - overall_start_time
        logger.error(f"JSON generation failed after {error_time:.2f}s: {e}")
        logger.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
