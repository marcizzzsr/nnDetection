import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import SimpleITK as sitk
from loguru import logger
from nndet.io import save_json
from nndet.io.itk import load_sitk
from nndet.utils.check import env_guard
from tqdm import tqdm


def prepare_case(
    case_id: str,
    images_dir: Path,
    labels_dir: Path,
    data_target: Path,
    label_target: Path,
    modalities: List[str] = ["t2w", "adc", "hbv"],
):
    """
    Prepare a single PI-CAI case for nnDetection.

    Args:
        case_id: Case identifier (e.g., "10000_1000000")
        images_dir: Directory containing image folders per patient
        labels_dir: Directory containing label files
        data_target: Target directory for processed images
        label_target: Target directory for processed labels
        modalities: List of modalities to include
    """
    start_time = time.time()

    try:
        # Extract patient ID from case_id (e.g., "10000" from "10000_1000000")
        patient_id = case_id.split("_")[0]
        patient_dir = images_dir / patient_id

        if not patient_dir.exists():
            logger.error(f"Patient directory not found: {patient_dir}")
            return False

        # Load modalities
        images = {}
        reference_image = None

        for modality in modalities:
            modality_file = patient_dir / f"{case_id}_{modality}.mha"

            if not modality_file.exists():
                logger.warning(
                    f"Missing modality {modality} for case {case_id}: {modality_file}"
                )
                return False

            image = load_sitk(str(modality_file))
            images[modality] = image

            # Use first modality as reference (typically t2w)
            if reference_image is None:
                reference_image = image

        # Resample all other modalities to reference space (t2w)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)

        for modality in modalities[1:]:  # Skip first (reference)
            if modality in images:
                images[modality] = resampler.Execute(images[modality])

        # Load and process label
        label_file = labels_dir / f"{case_id}.nii.gz"

        if not label_file.exists():
            logger.warning(f"Missing label for case {case_id} - creating empty label")
            # Create empty label for test cases
            mask_array = np.zeros(
                sitk.GetArrayFromImage(reference_image).shape, dtype=np.uint8
            )
            mask_image = sitk.GetImageFromArray(mask_array)
            mask_image.CopyInformation(reference_image)
            instances = {}
        else:
            # Load label and process
            mask_image = load_sitk(str(label_file))

            # Resample label to match reference image
            label_resampler = sitk.ResampleImageFilter()
            label_resampler.SetReferenceImage(reference_image)
            label_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            mask_image = label_resampler.Execute(mask_image)

            # Convert to binary mask and create instances
            mask_array = sitk.GetArrayFromImage(mask_image)

            # Find connected components for instance segmentation
            if np.any(mask_array > 0):
                # Convert to binary
                binary_mask = (mask_array > 0).astype(np.uint8)

                # Find connected components
                binary_image = sitk.GetImageFromArray(binary_mask)
                binary_image.CopyInformation(mask_image)

                cc_filter = sitk.ConnectedComponentImageFilter()
                cc_image = cc_filter.Execute(binary_image)
                cc_array = sitk.GetArrayFromImage(cc_image)

                # Create instances mapping (all lesions are csPCa = class 1)
                unique_labels = np.unique(cc_array)
                instances = {}
                for label_id in unique_labels:
                    if label_id > 0:  # Skip background
                        instances[int(label_id)] = 1  # All PI-CAI lesions are csPCa

                # Update mask with connected components
                mask_image = cc_image
            else:
                instances = {}

        # Save images with modality numbering
        for i, modality in enumerate(modalities):
            if modality in images:
                output_file = data_target / f"{case_id}_{i:04d}.nii.gz"
                sitk.WriteImage(images[modality], str(output_file))

        # Save label and instances
        sitk.WriteImage(mask_image, str(label_target / f"{case_id}.nii.gz"))
        save_json({"instances": instances}, label_target / f"{case_id}.json")

        processing_time = time.time() - start_time
        logger.info(
            f"{case_id} succesfully processed: {len(instances)} lesions in {processing_time:.2f}s"
        )
        return True

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"{case_id} failed after {processing_time:.2f}s: {e}")
        logger.debug(traceback.format_exc())
        return False


def discover_cases(
    images_dir: Path, labels_dir: Path, modalities: List[str] = ["t2w", "adc", "hbv"]
) -> List[str]:
    """
    Discover all valid PI-CAI cases from the dataset.

    Args:
        images_dir: Directory containing image folders per patient
        labels_dir: Directory containing label files
        modalities: Required modalities for a case to be valid

    Returns:
        List of valid case IDs
    """
    start_time = time.time()
    logger.info("Discovering PI-CAI cases...")

    valid_cases = []
    total_patients = 0
    total_case_candidates = 0

    # Scan patient directories
    patient_dirs = [d for d in images_dir.iterdir() if d.is_dir()]
    total_patients = len(patient_dirs)
    logger.info(f"Found {total_patients} patient directories")

    for patient_dir in tqdm(patient_dirs, desc="Scanning patients"):
        patient_id = patient_dir.name

        # Find all cases for this patient
        case_files = []
        for modality in modalities:
            case_files.extend(patient_dir.glob(f"*_{modality}.mha"))

        # Extract unique case IDs
        case_ids = set()
        for case_file in case_files:
            # Extract case_id from filename (remove modality suffix)
            case_id = case_file.stem.replace(f"_{case_file.stem.split('_')[-1]}", "")
            case_ids.add(case_id)

        total_case_candidates += len(case_ids)

        # Check if each case has all required modalities
        for case_id in case_ids:
            has_all_modalities = True
            missing_modalities = []

            for modality in modalities:
                modality_file = patient_dir / f"{case_id}_{modality}.mha"
                if not modality_file.exists():
                    has_all_modalities = False
                    missing_modalities.append(modality)

            if has_all_modalities:
                valid_cases.append(case_id)
            else:
                logger.debug(f"Skipping {case_id}: missing {missing_modalities}")

    discovery_time = time.time() - start_time
    logger.info(
        f"Discovery completed in {discovery_time:.2f}s: {len(valid_cases)}/{total_case_candidates} valid cases from {total_patients} patients"
    )
    return sorted(valid_cases)


def load_custom_splits(splits_file: Path) -> Dict[str, List[str]]:
    """Load custom train/test splits from JSON file."""
    logger.info(f"Loading custom splits from {splits_file}")

    try:
        with open(splits_file, "r") as f:
            splits_data = json.load(f)

        # Handle different split file formats
        if "test" in splits_data:
            test_cases = splits_data["test"]
            if "train" in splits_data:
                train_cases = splits_data["train"]
            else:
                # Assume all other cases are training (will be filtered later)
                train_cases = []
        else:
            raise ValueError("Splits file must contain 'test' key")

        logger.info(f"Loaded {len(test_cases)} test cases")
        if train_cases:
            logger.info(f"Loaded {len(train_cases)} train cases")
        else:
            logger.info(
                "No explicit train cases - will use remaining cases for training"
            )

        return {"train": train_cases, "test": test_cases}

    except Exception as e:
        logger.error(f"Failed to load splits file: {e}")
        raise


def create_splits_from_cases(
    all_cases: List[str], custom_splits: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """Create final train/test splits based on available cases and custom splits."""
    logger.info("Creating final train/test splits from available cases")

    test_cases = [case for case in custom_splits["test"] if case in all_cases]

    if custom_splits["train"]:
        # Use provided train cases
        train_cases = [case for case in custom_splits["train"] if case in all_cases]
    else:
        # Use all remaining cases as training
        train_cases = [case for case in all_cases if case not in test_cases]

    logger.info(f"Final splits: {len(train_cases)} train, {len(test_cases)} test")

    # Log missing cases
    missing_test = set(custom_splits["test"]) - set(test_cases)
    if missing_test:
        logger.warning(
            f"Missing test cases: {len(missing_test)} cases not found in dataset"
        )

    if custom_splits["train"]:
        missing_train = set(custom_splits["train"]) - set(train_cases)
        if missing_train:
            logger.warning(
                f"Missing train cases: {len(missing_train)} cases not found in dataset"
            )

    return {"train": train_cases, "test": test_cases}


@env_guard
def main():
    """
    Prepare PI-CAI dataset for nnDetection.
    Uses T2w, ADC, and HBV modalities with csPCa lesion annotations.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Prepare PI-CAI dataset for nnDetection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--splits_file",
        type=str,
        default="pre-process/data-split/test.json",
        help="Path to JSON file containing train/test splits (relative to project root or absolute path)",
    )
    parser.add_argument(
        "--source-image-dir",
        "-i",
        type=str,
        required=True,
        help="Directory of raw image data (expects only images inside specified folder)"
    )
    parser.add_argument(
        "--source-labels-dir",
        "-l",
        type=str,
        required=True,
        help="Directory of raw labels data (expects only images inside specified folder)"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="Path to output directory (will contain entire output structure)"
    )

    args = parser.parse_args()

    overall_start_time = time.time()

    task_data_dir = Path(args.output_dir) / "Task022_PICAI"
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
    logger.info("PI-CAI DATASET PREPARATION FOR NNDETECTION")
    logger.info("=" * 60)
    logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Setup raw paths
        # PI-CAI dataset structure
        images_dir = Path(args.source_image_dir)
        labels_dir = Path(args.source_labels_dir)

        if not images_dir.is_dir():
            raise RuntimeError(
                f"{images_dir} should contain the raw iamges but does not exist."
            )

        if not labels_dir.is_dir():
            raise RuntimeError(
                f"{labels_dir} should contain the raw labels but does not exist."
            )


        # Custom splits file - handle both relative and absolute paths
        splits_file = Path(args.splits_file)
        if not splits_file.is_absolute():
            logger.warning("Using a non-absolute path for splits file")

        logger.info(f"Using splits file: {splits_file}")

        # Validate input paths
        logger.info("Validating input paths...")
        if not images_dir.exists():
            raise RuntimeError(f"Images directory not found: {images_dir}")
        if not labels_dir.exists():
            raise RuntimeError(f"Labels directory not found: {labels_dir}")
        if not splits_file.exists():
            raise RuntimeError(f"Splits file not found: {splits_file}")
        logger.info("✓ All input paths validated")

        # Setup target paths
        logger.info("Creating output directories...")
        data_target = task_data_dir / "raw_splitted" / "imagesTr"
        data_target.mkdir(parents=True, exist_ok=True)
        label_target = task_data_dir / "raw_splitted" / "labelsTr"
        label_target.mkdir(parents=True, exist_ok=True)
        test_data_target = task_data_dir / "raw_splitted" / "imagesTs"
        test_data_target.mkdir(parents=True, exist_ok=True)
        logger.info("✓ Output directories created")

        # Configuration
        modalities = ["t2w", "adc", "hbv"]
        logger.info(f"Configuration: {modalities} modalities")

        # Save dataset metadata
        logger.info("Creating dataset metadata...")
        dataset_info = {
            "name": "PI-CAI",
            "task": "Task022_PICAI",
            "target_class": None,
            "test_labels": False,
            "labels": {"0": "background", "1": "csPCa_lesion"},
            "modalities": {"0": "T2w", "1": "ADC", "2": "HBV"},
            "dim": 3,
            "info": "PI-CAI dataset for prostate cancer detection and localization. Custom train/test splits applied.",
        }
        save_json(dataset_info, task_data_dir / "dataset.json")
        logger.info("✓ Dataset metadata saved")

        # Phase 1: Discover all valid cases
        logger.info("\n ------- Case Discovery")
        all_cases = discover_cases(images_dir, labels_dir, modalities)

        # Phase 2: Load custom splits
        logger.info("\n ------- Loading Custom Splits")
        custom_splits = load_custom_splits(splits_file)

        # Phase 3: Create final splits
        logger.info("\n ------- Creating Final Splits")
        splits = create_splits_from_cases(all_cases, custom_splits)
        save_json(splits, task_data_dir / "splits.json")

        # Phase 4: Process training cases
        logger.info("\n ------- Processing Training Cases")
        train_total = len(splits["train"])
        train_success = 0

        for i, case_id in enumerate(tqdm(splits["train"], desc="Training cases")):
            success = prepare_case(
                case_id=case_id,
                images_dir=images_dir,
                labels_dir=labels_dir,
                data_target=data_target,
                label_target=label_target,
                modalities=modalities,
            )
            if success:
                train_success += 1

        # Phase 5: Process test cases
        logger.info("\n🧪 PHASE 5: Processing Test Cases")
        test_total = len(splits["test"])
        test_success = 0

        for i, case_id in enumerate(tqdm(splits["test"], desc="Test cases")):
            success = prepare_case(
                case_id=case_id,
                images_dir=images_dir,
                labels_dir=labels_dir,
                data_target=test_data_target,
                label_target=label_target,  # Labels go to labelsTr even for test cases
                modalities=modalities,
            )
            if success:
                test_success += 1

        # Create nnDetection format splits
        final_splits = {
            "train": [f"{case_id}.nii.gz" for case_id in splits["train"]],
            "test": [f"{case_id}.nii.gz" for case_id in splits["test"]],
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
