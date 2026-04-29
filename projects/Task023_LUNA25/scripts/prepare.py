"""
LUNA25 nnDetection directory builder.

Reads the conversion log and lesion log produced by match_MedSAM_masks.py,
determines train/test splits from CSV files, and assembles the nnDetection-
compliant directory structure.  No matching logic is performed here.

Prerequisite: run match_MedSAM_masks.py first to produce the logs.

Outputs:
- Task023_LUNA25/raw_splitted/imagesTr, labelsTr, imagesTs, labelsTs
- Task023_LUNA25/dataset.json
- Task023_LUNA25/raw_splitted/splits.json
"""

import argparse
import json
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import SimpleITK as sitk
from loguru import logger
from tqdm import tqdm


def sanitize_uid(uid: str) -> str:
    return str(uid).replace(".", "_")


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def convert_mha_to_nifti(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    try:
        img = sitk.ReadImage(str(src))
        dst.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(img, str(dst))
        return True
    except Exception as e:
        logger.debug(f"Error converting {src} to {dst}: {e}")
        return False


def find_mask_for_uid(mask_dir: Path, uid: str) -> Optional[Path]:
    """
    Find mask for UID.
    Preference order:
    1) exact uid + .nii.gz
    2) exact uid + .nii
    3) sanitized uid + .nii.gz
    4) sanitized uid + .nii
    """
    uid_sanitized = sanitize_uid(uid)
    candidates = [
        mask_dir / f"{uid}.nii.gz",
        mask_dir / f"{uid}.nii",
        mask_dir / f"{uid_sanitized}.nii.gz",
        mask_dir / f"{uid_sanitized}.nii",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def build_instance_mapping_from_lesion_log(
    lesion_log: pd.DataFrame,
) -> Dict[str, Dict[str, int]]:
    """Build {uid: {instance_id_str: label}} from the lesion-level log."""
    matched = lesion_log[lesion_log["matched"] == True]  # noqa: E712
    mapping: Dict[str, Dict[str, int]] = {}
    for _, row in matched.iterrows():
        uid = str(row["uid"])
        inst_id = str(int(row["matched_instance_id"]))
        label = int(row["label"])
        mapping.setdefault(uid, {})[inst_id] = label
    return mapping


def process_case(
    uid: str,
    split_name: str,
    image_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    instance_mapping: Dict[str, int],
    labels_only: bool,
) -> Tuple[str, str, bool, Optional[str]]:
    """Copy mask + write JSON (+ convert image unless labels_only).

    Returns (uid, split_name, success, error_message).
    """
    uid_sanitized = sanitize_uid(uid)

    if split_name == "train":
        dst_img = output_dir / "imagesTr" / f"{uid_sanitized}_0000.nii.gz"
        dst_lbl = output_dir / "labelsTr" / f"{uid_sanitized}.nii.gz"
        dst_json = output_dir / "labelsTr" / f"{uid_sanitized}.json"
    else:
        dst_img = output_dir / "imagesTs" / f"{uid_sanitized}_0000.nii.gz"
        dst_lbl = output_dir / "labelsTs" / f"{uid_sanitized}.nii.gz"
        dst_json = output_dir / "labelsTs" / f"{uid_sanitized}.json"

    try:
        # Copy mask
        src_mask = find_mask_for_uid(mask_dir, uid)
        if src_mask is None:
            return uid, split_name, False, "Mask not found"
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_mask, dst_lbl)

        # Write instance mapping JSON
        save_json({"instances": instance_mapping}, dst_json)

        # Convert image
        if not labels_only:
            src_img = image_dir / f"{uid}.mha"
            if not convert_mha_to_nifti(src_img, dst_img):
                return uid, split_name, False, f"Image conversion failed for {src_img}"

        return uid, split_name, True, None
    except Exception as e:
        return uid, split_name, False, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build nnDetection directory structure for LUNA25 from pre-computed "
            "matching logs (produced by match_MedSAM_masks.py)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images", type=str, required=True, help="Path to LUNA25 images (.mha)"
    )
    parser.add_argument(
        "--masks", type=str, required=True, help="Path to MedSAM2 masks (.nii/.nii.gz)"
    )
    parser.add_argument(
        "--conversion-log",
        type=str,
        required=True,
        help="Path to case-level conversion log CSV (from match_MedSAM_masks.py)",
    )
    parser.add_argument(
        "--lesion-log",
        type=str,
        required=True,
        help="Path to lesion-level log CSV (from match_MedSAM_masks.py)",
    )
    parser.add_argument(
        "--train-csv", type=str, required=True, help="Path to luna25-train.csv"
    )
    parser.add_argument(
        "--test-csv", type=str, required=True, help="Path to luna25-test.csv"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to nnDetection raw data output root",
    )
    parser.add_argument(
        "--jobs", type=int, default=8, help="Number of parallel workers"
    )
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Only process labels (masks + JSON), skip image conversion",
    )
    parser.add_argument(
        "--single-class",
        action="store_true",
        help="Collapse benign/malignant into a single 'lesion' class (output to Task024_LUNA25_singleclass)",
    )
    args = parser.parse_args()

    overall_start_time = time.time()

    task_name = "Task024_LUNA25_singleclass" if args.single_class else "Task023_LUNA25"
    task_data_dir = Path(args.output) / task_name
    raw_splitted_dir = task_data_dir / "raw_splitted"

    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")
    logger.add(
        task_data_dir / "prepare.log",
        format="{time} | {level} | {message}",
        level="DEBUG",
    )

    logger.info("=" * 60)
    logger.info("LUNA25 NNDETECTION DIRECTORY BUILDER")
    logger.info("=" * 60)
    logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        image_dir = Path(args.images)
        mask_dir = Path(args.masks)

        # ---- validate inputs ------------------------------------------------
        for label, p in [
            ("Images", image_dir),
            ("Masks", mask_dir),
            ("Conversion log", Path(args.conversion_log)),
            ("Lesion log", Path(args.lesion_log)),
            ("Train CSV", Path(args.train_csv)),
            ("Test CSV", Path(args.test_csv)),
        ]:
            if not p.exists():
                raise RuntimeError(f"{label} not found: {p}")
        logger.info("All input paths validated")

        # ---- read CSVs ------------------------------------------------------
        conversion_log = pd.read_csv(args.conversion_log)
        lesion_log = pd.read_csv(args.lesion_log)
        train_df = pd.read_csv(args.train_csv)
        test_df = pd.read_csv(args.test_csv)

        train_uids = set(train_df["SeriesInstanceUID"].astype(str).unique())
        test_uids = set(test_df["SeriesInstanceUID"].astype(str).unique())
        logger.info(f"Split sizes: {len(train_uids)} train, {len(test_uids)} test")

        # ---- build instance mappings from lesion log ------------------------
        instance_mappings = build_instance_mapping_from_lesion_log(lesion_log)

        if args.single_class:
            for uid_map in instance_mappings.values():
                for inst_id in uid_map:
                    uid_map[inst_id] = 0
            logger.info(
                "Single-class mode: all instance labels remapped to 0 ('lesion')"
            )

        # ---- filter conversion log to non-discarded successes ---------------
        valid_mask = (conversion_log["status"] == "success") & (
            conversion_log["discarded"] != True  # noqa: E712
        )
        valid_cases = conversion_log.loc[valid_mask]
        logger.info(
            f"Conversion log: {len(conversion_log)} total, {len(valid_cases)} valid"
        )

        # ---- create output directories --------------------------------------
        for subdir in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
            (raw_splitted_dir / subdir).mkdir(parents=True, exist_ok=True)

        # ---- process cases in parallel --------------------------------------
        training_entries: List[str] = []
        test_entries: List[str] = []
        train_success = 0
        test_success = 0
        skipped_no_split = 0

        futures = []
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            for _, row in valid_cases.iterrows():
                uid = str(row["uid"])
                if uid in train_uids:
                    split_name = "train"
                elif uid in test_uids:
                    split_name = "test"
                else:
                    skipped_no_split += 1
                    logger.debug(f"UID {uid} not in train or test CSV, skipping")
                    continue

                mapping = instance_mappings.get(uid, {})
                futures.append(
                    executor.submit(
                        process_case,
                        uid=uid,
                        split_name=split_name,
                        image_dir=image_dir,
                        mask_dir=mask_dir,
                        output_dir=raw_splitted_dir,
                        instance_mapping=mapping,
                        labels_only=args.labels_only,
                    )
                )

            for future in tqdm(futures, desc="Building dataset"):
                uid, split_name, success, error = future.result()
                if success:
                    entry = f"{sanitize_uid(uid)}.nii.gz"
                    if split_name == "train":
                        training_entries.append(entry)
                        train_success += 1
                    else:
                        test_entries.append(entry)
                        test_success += 1
                else:
                    logger.warning(f"Skipping {split_name} case {uid}: {error}")

        # ---- write metadata -------------------------------------------------
        dataset_info = {
            "name": "LUNA25",
            "task": task_name,
            "target_class": None if args.single_class else 1,
            "test_labels": True,
            "labels": {"0": "lesion"}
            if args.single_class
            else {"0": "benign", "1": "malignant"},
            "modalities": {"0": "CT"},
            "dim": 3,
            "info": "LUNA25 dataset for lung nodule detection and classification.",
        }
        save_json(dataset_info, task_data_dir / "dataset.json")
        save_json(
            {"train": training_entries, "test": test_entries},
            raw_splitted_dir / "splits.json",
        )

        # ---- summary --------------------------------------------------------
        overall_time = time.time() - overall_start_time
        logger.info("=" * 60)
        logger.info("PREPARATION COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Training: {train_success} cases")
        logger.info(f"Test:     {test_success} cases")
        if skipped_no_split:
            logger.info(f"Skipped (not in train/test CSV): {skipped_no_split}")
        logger.info(
            f"Total time: {overall_time:.2f}s ({overall_time / 60:.1f} minutes)"
        )
        logger.info(f"Output: {task_data_dir}")

    except Exception as e:
        error_time = time.time() - overall_start_time
        logger.error(f"Preparation failed after {error_time:.2f}s: {e}")
        logger.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
