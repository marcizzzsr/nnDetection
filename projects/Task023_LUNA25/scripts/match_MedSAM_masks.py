"""
Standalone LUNA25 preparation script for nnDetection.

This script combines:
1) Full dataset preparation flow from prepare.py
2) Robust MedSAMv2 instance mapping diagnostics from prepare_MedSAMv2_masks.py

Outputs:
- Task023_LUNA25/raw_splitted/imagesTr, labelsTr, imagesTs, labelsTs
- Task023_LUNA25/dataset.json
- Task023_LUNA25/raw_splitted/splits.json
- Task023_LUNA25/conversion_log.csv
- Task023_LUNA25/medsam_lesion_match_report.csv
- Task023_LUNA25/prepare.log
"""

import argparse
import json
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import nibabel as nib
import numpy as np
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


def _in_bounds(x: int, y: int, z: int, shape: Tuple[int, int, int]) -> bool:
    return 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]


def _find_instance_id_with_local_search(
    data: np.ndarray,
    x: int,
    y: int,
    z: int,
    radius: int,
) -> Tuple[int, bool, Optional[Tuple[int, int, int]]]:
    """
    Return (instance_id, used_local_search, matched_voxel).
    """
    nid = int(data[x, y, z])
    if nid > 0:
        return nid, False, (x, y, z)

    if radius <= 0:
        return 0, False, None

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                xx, yy, zz = x + dx, y + dy, z + dz
                if _in_bounds(xx, yy, zz, data.shape):
                    nid = int(data[xx, yy, zz])
                    if nid > 0:
                        return nid, True, (xx, yy, zz)

    return 0, False, None


def build_instance_mapping_with_diagnostics(
    data: np.ndarray,
    affine: np.ndarray,
    case_rows: pd.DataFrame,
    local_search_radius: int,
) -> Dict[str, object]:
    inv_affine = np.linalg.inv(affine)

    unique_ids = np.unique(data).astype(np.int64)
    unique_ids = unique_ids[unique_ids > 0]
    unique_id_set: Set[int] = set(unique_ids.tolist())

    instance_mapping: Dict[str, int] = {}
    matched_ids: Set[int] = set()

    out_of_bounds = 0
    background_hits = 0
    native_hits = 0
    flipped_xy_hits = 0
    local_search_hits = 0
    lesion_rows: List[Dict[str, object]] = []

    optional_id_fields = [
        "PatientID",
        "StudyDate",
        "LesionID",
        "AnnotationID",
        "NoduleID",
    ]

    for _, row in case_rows.iterrows():
        xw = float(row["CoordX"])
        yw = float(row["CoordY"])
        zw = float(row["CoordZ"])

        # Try both conventions: native and LPS->RAS flipped XY.
        candidate_world_coords = [
            np.array([xw, yw, zw, 1.0]),
            np.array([-xw, -yw, zw, 1.0]),
        ]

        found = False
        any_in_bounds = False
        matched_instance_id = None
        matched_voxel = None
        query_voxel = None
        used_local_search_flag = False
        used_convention = ""
        used_world_xyz = None

        for candidate_idx, world_coords in enumerate(candidate_world_coords):
            voxel_coords = inv_affine @ world_coords
            x, y, z = np.round(voxel_coords[:3]).astype(int)

            if not _in_bounds(x, y, z, data.shape):
                continue

            any_in_bounds = True
            lesion_id, used_local_search, matched_voxel_candidate = (
                _find_instance_id_with_local_search(
                    data=data,
                    x=x,
                    y=y,
                    z=z,
                    radius=local_search_radius,
                )
            )
            query_voxel = (int(x), int(y), int(z))

            if lesion_id > 0:
                instance_mapping[str(lesion_id)] = int(row["label"])
                matched_ids.add(lesion_id)
                matched_instance_id = int(lesion_id)
                matched_voxel = matched_voxel_candidate
                used_local_search_flag = bool(used_local_search)
                used_world_xyz = (
                    float(world_coords[0]),
                    float(world_coords[1]),
                    float(world_coords[2]),
                )

                if candidate_idx == 0:
                    native_hits += 1
                    used_convention = "native"
                else:
                    flipped_xy_hits += 1
                    used_convention = "flipped_xy"

                if used_local_search:
                    local_search_hits += 1

                found = True
                break

        if not found:
            if not any_in_bounds:
                out_of_bounds += 1
            else:
                background_hits += 1

        lesion_report_row: Dict[str, object] = {
            "SeriesInstanceUID": str(row["SeriesInstanceUID"]),
            "CoordX": xw,
            "CoordY": yw,
            "CoordZ": zw,
            "label": int(row["label"]),
            "matched": bool(found),
            "matched_instance_id": matched_instance_id,
            "match_convention": used_convention if found else "unmatched",
            "local_search_used": bool(used_local_search_flag),
            "search_radius": local_search_radius,
            "in_bounds": bool(any_in_bounds),
            "query_voxel_x": query_voxel[0] if query_voxel else None,
            "query_voxel_y": query_voxel[1] if query_voxel else None,
            "query_voxel_z": query_voxel[2] if query_voxel else None,
            "matched_voxel_x": matched_voxel[0] if matched_voxel else None,
            "matched_voxel_y": matched_voxel[1] if matched_voxel else None,
            "matched_voxel_z": matched_voxel[2] if matched_voxel else None,
            "distance_voxel": None,
            "distance_mm": None,
        }

        for field in optional_id_fields:
            lesion_report_row[field] = row[field] if field in row.index else None

        if (
            found
            and matched_voxel is not None
            and query_voxel is not None
            and used_world_xyz is not None
        ):
            qx, qy, qz = query_voxel
            mx, my, mz = matched_voxel
            lesion_report_row["distance_voxel"] = float(
                np.linalg.norm(np.array([mx - qx, my - qy, mz - qz], dtype=np.float64))
            )
            matched_world = affine @ np.array([mx, my, mz, 1.0], dtype=np.float64)
            lesion_report_row["distance_mm"] = float(
                np.linalg.norm(
                    np.array(
                        [
                            matched_world[0] - used_world_xyz[0],
                            matched_world[1] - used_world_xyz[1],
                            matched_world[2] - used_world_xyz[2],
                        ],
                        dtype=np.float64,
                    )
                )
            )

        lesion_rows.append(lesion_report_row)

    missing_ids = sorted(unique_id_set - matched_ids)

    return {
        "instance_mapping": instance_mapping,
        "unique_id_set": unique_id_set,
        "matched_ids": matched_ids,
        "missing_ids": missing_ids,
        "unique_mask_ids_count": int(len(unique_id_set)),
        "mapped_ids_count": int(len(instance_mapping)),
        "missing_ids_count": int(len(missing_ids)),
        "out_of_bounds_count": int(out_of_bounds),
        "background_hit_count": int(background_hits),
        "native_hit_count": int(native_hits),
        "flipped_xy_hit_count": int(flipped_xy_hits),
        "local_search_hit_count": int(local_search_hits),
        "matched_instance_ids": " ".join(map(str, sorted(matched_ids))),
        "missing_instance_ids": " ".join(map(str, missing_ids)),
        "lesion_rows": lesion_rows,
    }


def process_case(
    uid: str,
    mask_dir: Path,
    annotations_group: pd.DataFrame,
    local_search_radius: int,
    strict_instance_match: bool,
    discard_duplicates: bool,
) -> Tuple[str, bool, Optional[str], Dict[str, object], List[Dict[str, object]]]:
    uid_sanitized = sanitize_uid(uid)

    log: Dict[str, object] = {
        "uid": uid,
        "uid_sanitized": uid_sanitized,
        "status": "success",
        "error_message": "",
        "mask_path": "",
        "csv_rows_count": int(len(annotations_group)),
        "local_search_radius": local_search_radius,
        "unique_mask_ids_count": 0,
        "mapped_ids_count": 0,
        "missing_ids_count": 0,
        "out_of_bounds_count": 0,
        "background_hit_count": 0,
        "native_hit_count": 0,
        "flipped_xy_hit_count": 0,
        "local_search_hit_count": 0,
        "matched_instance_ids": "",
        "missing_instance_ids": "",
        "unmatched_lesion_count": 0,
        "has_duplicates_matching": False,
        "discarded": False,
        "json_path": "",
    }

    try:
        src_mask = find_mask_for_uid(mask_dir, uid)
        if src_mask is None:
            msg = "Mask not found"
            log["status"] = "mask_not_found"
            log["error_message"] = msg
            return uid, False, msg, log, []
        log["mask_path"] = str(src_mask)

        nii = nib.load(src_mask)
        data = nii.get_fdata()
        mapping_diag = build_instance_mapping_with_diagnostics(
            data=data,
            affine=nii.affine,
            case_rows=annotations_group,
            local_search_radius=local_search_radius,
        )

        log["unique_mask_ids_count"] = mapping_diag["unique_mask_ids_count"]
        log["mapped_ids_count"] = mapping_diag["mapped_ids_count"]
        log["missing_ids_count"] = mapping_diag["missing_ids_count"]
        log["out_of_bounds_count"] = mapping_diag["out_of_bounds_count"]
        log["background_hit_count"] = mapping_diag["background_hit_count"]
        log["native_hit_count"] = mapping_diag["native_hit_count"]
        log["flipped_xy_hit_count"] = mapping_diag["flipped_xy_hit_count"]
        log["local_search_hit_count"] = mapping_diag["local_search_hit_count"]
        log["matched_instance_ids"] = mapping_diag["matched_instance_ids"]
        log["missing_instance_ids"] = mapping_diag["missing_instance_ids"]
        lesion_rows = mapping_diag["lesion_rows"]

        # Compute diagnostic fields (always, regardless of flags)
        unmatched_lesion_count = sum(1 for r in lesion_rows if not r["matched"])
        matched_instance_ids_list = [
            r["matched_instance_id"] for r in lesion_rows if r["matched"]
        ]
        has_duplicates_matching = len(matched_instance_ids_list) != len(
            set(matched_instance_ids_list)
        )
        log["unmatched_lesion_count"] = unmatched_lesion_count
        log["has_duplicates_matching"] = has_duplicates_matching

        if len(annotations_group) == 0:
            log["status"] = "no_csv_match"

        if strict_instance_match and unmatched_lesion_count > 0:
            msg = (
                f"Unmatched lesions: {unmatched_lesion_count} of "
                f"{len(annotations_group)} lesions not matched"
            )
            log["status"] = "discarded_unmatched_lesions"
            log["error_message"] = msg
            log["discarded"] = True
            return uid, False, msg, log, lesion_rows

        if discard_duplicates and has_duplicates_matching:
            msg = (
                "Duplicate instance mapping: multiple lesions mapped "
                "to the same mask instance ID"
            )
            log["status"] = "discarded_duplicate_instances"
            log["error_message"] = msg
            log["discarded"] = True
            return uid, False, msg, log, lesion_rows

        if log["status"] == "no_csv_match":
            return uid, False, str(log["status"]), log, lesion_rows

        return uid, True, None, log, lesion_rows

    except Exception as e:
        msg = f"Failed to process case: {e}"
        log["status"] = "error"
        log["error_message"] = msg
        return uid, False, msg, log, []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone LUNA25 preparation for nnDetection (fused prepare + MedSAMv2 diagnostics)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to original LUNA25 images (.mha)",
    )
    parser.add_argument(
        "--masks", type=str, required=True, help="Path to MedSAM2 masks (.nii/.nii.gz)"
    )

    parser.add_argument(
        "-c", "--csv", type=str, required=True, help="Path to annotations CSV"
    )
    parser.add_argument(
        "--jobs", type=int, default=8, help="Number of parallel workers"
    )
    parser.add_argument(
        "--local-search-radius",
        type=int,
        default=1,
        help="Local neighborhood search radius around rounded voxel coordinates",
    )
    parser.add_argument(
        "--strict-instance-match",
        action="store_true",
        default=False,
        help="Discard cases with unmatched lesions (default: disabled)",
    )

    parser.add_argument(
        "--discard-duplicates",
        action="store_true",
        default=False,
        help="Discard cases where multiple lesions map to the same mask instance ID (default: disabled)",
    )

    parser.add_argument(
        "--case-log-csv",
        type=str,
        default=None,
        help="Path to case-level conversion log CSV. Defaults to <masks>/matching_case_log.csv",
    )

    parser.add_argument(
        "--lesion-log-csv",
        dest="lesion_report_csv",
        type=str,
        default=None,
        help="Path to lesion-level report CSV. Defaults to <masks>/matching_lesion_log.csv",
    )


    args = parser.parse_args()

    overall_start_time = time.time()

    task_data_dir = Path(args.output) / "Task023_LUNA25"
    raw_splitted_dir = task_data_dir / "raw_splitted"
    if not args.report_only:
        raw_splitted_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}", level="INFO")
    if not args.report_only:
        logger.add(
            task_data_dir / "prepare.log",
            format="{time} | {level} | {message}",
            level="DEBUG",
        )

    logger.info("=" * 60)
    logger.info("LUNA25 DATASET PREPARATION FOR NNDETECTION (FUSED)")
    logger.info("=" * 60)
    logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Local search radius: {args.local_search_radius}")

    try:
        image_dir = Path(args.images)
        mask_dir = Path(args.masks)

        logger.info("Validating input paths...")
        if not image_dir.exists():
            raise RuntimeError(f"Images directory not found: {image_dir}")
        if not mask_dir.exists():
            raise RuntimeError(f"Masks directory not found: {mask_dir}")
        logger.info("All input paths validated")

        logger.info("Loading annotations...")
        annotations_csv = pd.read_csv(args.csv)

        lesion_report_csv_path = (
            Path(args.lesion_report_csv)
            if args.lesion_report_csv
            else mask_dir / "lesion_match_report.csv"
        )
        lesion_report_csv_path.parent.mkdir(parents=True, exist_ok=True)

        required_cols = ["SeriesInstanceUID", "CoordX", "CoordY", "CoordZ", "label"]
        for col in required_cols:
            if col not in annotations_csv.columns:
                raise ValueError(f"Column '{col}' not found in annotations CSV")

        instances = pd.unique(annotations_csv["SeriesInstanceUID"])

        logger.info(
            f"Found {len(instances)} instances"
        )

        training_entries: List[str] = []
        conversion_logs: List[Dict[str, object]] = []
        lesion_report_rows: List[Dict[str, object]] = []

        logger.info(f"Processing instance cases with {args.jobs} workers")
        match_success = 0
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = []
            for uid in instances:
                annotations = annotations_csv[
                    annotations_csv["SeriesInstanceUID"].astype(str) == str(uid)
                ]
                futures.append(
                    executor.submit(
                        process_case,
                        uid=str(uid),
                        image_dir=image_dir,
                        mask_dir=mask_dir,
                        annotations_group=annotations,
                        local_search_radius=args.local_search_radius,
                        strict_instance_match=args.strict_instance_match,
                        discard_duplicates=args.discard_duplicates,
                    )
                )

            for future in tqdm(futures, total=len(instances), desc="Instance cases"):
                uid, success, error, case_log, case_lesions = future.result()
                conversion_logs.append(case_log)
                lesion_report_rows.extend(
                    [
                        {
                            "uid": uid,
                            "uid_sanitized": sanitize_uid(uid),
                            **row,
                        }
                        for row in case_lesions
                    ]
                )
                if success:
                    training_entries.append(f"{sanitize_uid(uid)}.nii.gz")
                    match_success += 1
                else:
                    logger.warning(f"Skipping instance case {uid}: {error}")

        lesion_report_df = pd.DataFrame(lesion_report_rows)
        lesion_report_df.to_csv(lesion_report_csv_path, index=False)


        log_df = pd.DataFrame(conversion_logs)
        if not args.report_only:
            log_csv_path = (
                Path(args.case_log_csv)
                if args.case_log_csv
                else mask_dir / "conversion_log.csv"
            )
            log_df.to_csv(log_csv_path, index=False)

        overall_time = time.time() - overall_start_time
        logger.info("=" * 60)
        logger.info("MATCHING COMPLETED")
        logger.info("=" * 60)
        logger.info(
            f"Matching success: {match_success}/{len(instances)} cases ({(100.0 * match_success / max(1, len(instances))):.1f}%)"
        )

        if not log_df.empty:
            logger.info(
                "Detailed mapping summary: "
                f"success={(log_df['status'] == 'success').sum()} "
                f"no_csv_match={(log_df['status'] == 'no_csv_match').sum()} "
                f"discarded_unmatched={(log_df['status'] == 'discarded_unmatched_lesions').sum()} "
                f"discarded_duplicates={(log_df['status'] == 'discarded_duplicate_instances').sum()} "
                f"error={(log_df['status'] == 'error').sum()}"
            )
        logger.info(f"Lesion-level report saved to: {lesion_report_csv_path}")
        if not args.report_only:
            logger.info(f"Conversion log saved to: {log_csv_path}")
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
