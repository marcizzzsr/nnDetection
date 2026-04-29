"""
Compute NLST-compliant nodule diameters from segmentation masks.

For each matched lesion, identifies the axial slice with the largest
cross-sectional area, fits the 2D contour in physical (mm) space using PCA,
and computes the NLST diameter as the mean of the two orthogonal diameters:

    nlst_diameter_mm = (D1 + D2) / 2

where D1 is the longest in-plane diameter and D2 is the longest diameter
perpendicular to D1.

Prerequisite: run match_MedSAM_masks.py first to produce the logs.

Outputs:
- lesion_nlst_diameters.csv  (one row per matched lesion)
"""

import argparse
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from loguru import logger
from skimage.measure import find_contours
from tqdm import tqdm


def compute_nlst_diameter_for_row(row: dict, mask_dir: Path) -> dict:
    uid = row["uid"]
    instance_id = int(row["matched_instance_id"])
    img_path = mask_dir / (uid + ".nii.gz")

    nan_result = {**row, "nlst_diameter_mm": np.nan}

    try:
        sitk_image = sitk.ReadImage(str(img_path))
        # GetSpacing() -> (sx, sy, sz): x=col direction, y=row direction
        sx, sy = sitk_image.GetSpacing()[:2]

        data = sitk.GetArrayViewFromImage(sitk_image)  # (z, y, x)

        # Isolate this specific lesion by instance ID (no CC step needed)
        binary_mask = (data == instance_id).astype(np.uint8)
        if binary_mask.sum() == 0:
            return nan_result

        # Find the axial slice with the largest cross-sectional area
        z_best = int(np.argmax(binary_mask.sum(axis=(1, 2))))
        slice_2d = binary_mask[z_best]  # (ny, nx)

        # Extract 2D contour; select the longest one
        contours = find_contours(slice_2d, level=0.5)
        if not contours:
            return nan_result

        contour = max(contours, key=len)  # (N, 2) in (row, col) = (y, x) order
        if len(contour) < 2:
            return nan_result

        # Convert pixel coordinates to physical mm:
        # contour[:, 0] = row index (y direction) -> multiply by sy
        # contour[:, 1] = col index (x direction) -> multiply by sx
        pts_mm = contour * np.array([sy, sx])

        # PCA via SVD to find principal axes in-plane
        centered = pts_mm - pts_mm.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)  # Vt: (2, 2)
        proj = centered @ Vt.T  # (N, 2): projections onto each principal axis

        D1 = float(proj[:, 0].max() - proj[:, 0].min())  # span along 1st axis
        D2 = float(proj[:, 1].max() - proj[:, 1].min())  # span along 2nd axis

        return {**row, "nlst_diameter_mm": (D1 + D2) / 2.0}

    except Exception as e:
        logger.debug(f"Error processing {uid} instance {instance_id}: {e}")
        return nan_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute NLST-compliant nodule diameters from segmentation masks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--masks",
        type=str,
        required=True,
        help="Path to MedSAM masks directory (.nii.gz)",
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
        "--output",
        type=str,
        default="lesion_nlst_diameters.csv",
        help="Output CSV path",
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
    logger.info("LUNA25 NLST DIAMETER COMPUTATION")
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

        # ---- compute diameters in parallel ----------------------------------
        rows_list = [row.to_dict() for _, row in matched.iterrows()]

        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = executor.map(
                lambda r: compute_nlst_diameter_for_row(r, mask_dir), rows_list
            )
            results = list(
                tqdm(futures, total=len(rows_list), desc="Computing NLST diameters")
            )

        diameters_df = pd.DataFrame(results)

        # ---- summary --------------------------------------------------------
        n_valid = diameters_df["nlst_diameter_mm"].notna().sum()
        logger.info(f"Total lesions processed: {len(diameters_df)}")
        logger.info(f"Lesions with valid diameter: {n_valid}")
        if n_valid > 0:
            logger.info(
                f"Mean NLST diameter:   {diameters_df['nlst_diameter_mm'].mean():.2f} mm"
            )
            logger.info(
                f"Median NLST diameter: {diameters_df['nlst_diameter_mm'].median():.2f} mm"
            )

        # ---- save -----------------------------------------------------------
        diameters_df.to_csv(output_path, index=False)
        logger.info(f"Saved: {output_path}")

        overall_time = time.time() - overall_start_time
        logger.info(
            f"Total time: {overall_time:.2f}s ({overall_time / 60:.1f} minutes)"
        )

    except Exception as e:
        error_time = time.time() - overall_start_time
        logger.error(f"Computation failed after {error_time:.2f}s: {e}")
        logger.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
