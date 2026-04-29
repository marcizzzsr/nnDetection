"""
Compute Wadell's sphericity for each matched MedSAMv2 lesion mask.

Reads the conversion log and lesion-level match report, loads each NIfTI
mask, extracts the per-instance binary volume, and computes sphericity via
pyradiomics shape features on an isotropic resampled mask.

Prerequisite: run match_MedSAM_masks.py first to produce the logs.

Outputs:
- lesion_sphericity.csv  (one row per matched lesion)
"""

import argparse
import logging
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from loguru import logger
from radiomics import featureextractor
from tqdm import tqdm

# Suppress verbose pyradiomics output
logging.getLogger("radiomics").setLevel(logging.ERROR)

_SPHERICITY_KEY = "original_shape_Sphericity"
_SURFACE_AREA_KEY = "original_shape_SurfaceArea"
_MESH_VOLUME_KEY = "original_shape_MeshVolume"
_VOXEL_VOLUME_KEY = "original_shape_VoxelVolume"


def build_extractor() -> featureextractor.RadiomicsFeatureExtractor:
    """Configure a pyradiomics extractor with isotropic resampling and shape features only."""
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings.update({
        "resampledPixelSpacing": [1.0, 1.0, 1.0],
        "interpolator": "sitkLinear",
    })
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("shape")
    return extractor


def compute_sphericity_for_row(
    row: dict,
    mask_dir: Path,
    extractor: featureextractor.RadiomicsFeatureExtractor,
) -> dict:
    uid = row["uid"]
    instance_id = int(row["matched_instance_id"])
    img_path = mask_dir / (uid + ".nii.gz")

    try:
        sitk_image = sitk.ReadImage(str(img_path))
        data = sitk.GetArrayViewFromImage(sitk_image)  # (z, y, x)

        binary_mask = (data == instance_id).astype(np.uint8)
        if binary_mask.sum() == 0:
            return {**row, "sphericity": np.nan, "surface_area_mm2": np.nan,
                    "n_voxels": 0, "voxels_volume": np.nan, "mesh_volume_mm3": np.nan}

        sitk_mask = sitk.GetImageFromArray(binary_mask)
        sitk_mask.CopyInformation(sitk_image)

        # Shape features do not use intensity; cast to float for a valid image type
        sitk_float = sitk.Cast(sitk_image, sitk.sitkFloat32)

        result = extractor.execute(sitk_float, sitk_mask, label=1)

        voxel_volume = float(result[_VOXEL_VOLUME_KEY])

        # Minimum size check in isotropic space (< 20 mm^3 ≈ < 20 isotropic voxels)
        if voxel_volume < 20.0:
            return {**row, "sphericity": np.nan, "surface_area_mm2": np.nan,
                    "n_voxels": int(round(voxel_volume)), "voxels_volume": voxel_volume,
                    "mesh_volume_mm3": np.nan}

        sphericity = float(result[_SPHERICITY_KEY])
        surface_area = float(result[_SURFACE_AREA_KEY])
        mesh_volume = float(result[_MESH_VOLUME_KEY])

        if np.isfinite(sphericity) and 1.0 < sphericity <= 1.001:
            sphericity = 1.0

        return {
            **row,
            "sphericity": sphericity,
            "surface_area_mm2": surface_area,
            "n_voxels": int(round(voxel_volume)),
            "voxels_volume": voxel_volume,
            "mesh_volume_mm3": mesh_volume,
        }

    except Exception as e:
        logger.debug(f"Error processing {uid} instance {instance_id}: {e}")
        return {**row, "sphericity": np.nan, "surface_area_mm2": np.nan,
                "n_voxels": np.nan, "voxels_volume": np.nan, "mesh_volume_mm3": np.nan}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Wadell's sphericity for each matched MedSAMv2 lesion mask."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--masks", type=str, required=True, help="Path to MedSAM masks directory (.nii.gz)"
    )
    parser.add_argument(
        "--conversion-log", type=str, required=True,
        help="Path to case-level conversion log CSV (from match_MedSAM_masks.py)",
    )
    parser.add_argument(
        "--lesion-log", type=str, required=True,
        help="Path to lesion-level log CSV (from match_MedSAM_masks.py)",
    )
    parser.add_argument(
        "--output", type=str, default="lesion_sphericity.csv", help="Output CSV path",
    )
    parser.add_argument(
        "--jobs", type=int, default=8, help="Number of parallel workers"
    )
    args = parser.parse_args()

    overall_start_time = time.time()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")

    logger.info("=" * 60)
    logger.info("LUNA25 SPHERICITY COMPUTATION")
    logger.info("=" * 60)

    try:
        mask_dir = Path(args.masks)

        # ---- validate inputs ------------------------------------------------
        for label, p in [
            ("Masks", mask_dir),
            ("Conversion log", Path(args.conversion_log)),
            ("Lesion log", Path(args.lesion_log)),
        ]:
            if not p.exists():
                raise RuntimeError(f"{label} not found: {p}")
        logger.info("All input paths validated")

        # ---- read CSVs ------------------------------------------------------
        conversion_log = pd.read_csv(args.conversion_log)
        lesion_log = pd.read_csv(args.lesion_log)

        # ---- filter to valid, matched lesions -------------------------------
        valid_uids = set(
            conversion_log.loc[
                conversion_log["discarded"] == False, "uid"  # noqa: E712
            ]
            .astype(str)
            .unique()
        )
        matched = lesion_log[
            (lesion_log["uid"].astype(str).isin(valid_uids))
            & (lesion_log["matched_instance_id"].notna())
        ].copy()

        logger.info(
            f"Conversion log: {len(conversion_log)} total, {len(valid_uids)} non-discarded UIDs"
        )
        logger.info(
            f"Lesion log: {len(lesion_log)} total, {len(matched)} matched lesions to process"
        )

        # ---- build extractor once, shared across all workers ----------------
        extractor = build_extractor()
        logger.info("Pyradiomics extractor ready (shape features, 1mm isotropic resampling)")

        # ---- compute sphericity in parallel ---------------------------------
        rows_list = [row.to_dict() for _, row in matched.iterrows()]

        results = []
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = executor.map(
                lambda r: compute_sphericity_for_row(r, mask_dir, extractor), rows_list
            )
            results = list(tqdm(futures, total=len(rows_list), desc="Computing sphericity"))

        sphericity_df = pd.DataFrame(results)

        # ---- summary --------------------------------------------------------
        n_valid = sphericity_df["sphericity"].notna().sum()
        logger.info(f"Total lesions processed: {len(sphericity_df)}")
        logger.info(f"Lesions with valid sphericity: {n_valid}")
        if n_valid > 0:
            logger.info(f"Mean sphericity: {sphericity_df['sphericity'].mean():.4f}")
            logger.info(f"Median sphericity: {sphericity_df['sphericity'].median():.4f}")

        # ---- save -----------------------------------------------------------
        sphericity_df.to_csv(output_path, index=False)
        logger.info(f"Saved: {output_path}")

        overall_time = time.time() - overall_start_time
        logger.info(f"Total time: {overall_time:.2f}s ({overall_time / 60:.1f} minutes)")

    except Exception as e:
        error_time = time.time() - overall_start_time
        logger.error(f"Computation failed after {error_time:.2f}s: {e}")
        logger.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()