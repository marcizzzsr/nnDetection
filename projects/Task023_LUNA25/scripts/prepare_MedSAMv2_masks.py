"""
The following script prepares masks created using MedSAMv2 to match
nnDetection conventions.
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


def sanitize_uid(uid: str) -> str:
    return str(uid).replace(".", "_")


def case_id_from_path(mask_path: Path) -> str:
    name = mask_path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    raise ValueError(f"Unsupported mask extension for {mask_path}")


def discover_masks(mask_dir: Path) -> List[Tuple[str, Path]]:
    """
    Discover masks from the filesystem and prefer .nii.gz when both exist.
    Returns: list of (case_id, mask_path)
    """
    preferred: Dict[str, Path] = {}

    for p in sorted(mask_dir.glob("*.nii.gz")):
        preferred[case_id_from_path(p)] = p

    for p in sorted(mask_dir.glob("*.nii")):
        case_id = case_id_from_path(p)
        if case_id not in preferred:
            preferred[case_id] = p

    return sorted(preferred.items(), key=lambda x: x[0])


def process_case_with_logging(
    case_id: str,
    mask_path: Path,
    case_rows: pd.DataFrame,
    output_dir: Path,
    json_only: bool,
) -> Dict[str, object]:
    log = {
        "case_id": case_id,
        "mask_path": str(mask_path),
        "status": "success",
        "csv_rows_count": int(len(case_rows)),
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
        "json_path": "",
        "error_message": "",
    }

    try:
        nii = nib.load(mask_path)
        data = nii.get_fdata()
        affine = nii.affine
        inv_affine = np.linalg.inv(affine)

        unique_ids = np.unique(data).astype(np.int64)
        unique_ids = unique_ids[unique_ids > 0]
        unique_id_set = set(unique_ids.tolist())
        log["unique_mask_ids_count"] = int(len(unique_id_set))

        instance_mapping: Dict[str, int] = {}
        matched_ids = set()
        out_of_bounds = 0
        background_hits = 0
        native_hits = 0
        flipped_xy_hits = 0
        local_search_hits = 0

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

            for candidate_idx, world_coords in enumerate(candidate_world_coords):
                voxel_coords = inv_affine @ world_coords
                x, y, z = np.round(voxel_coords[:3]).astype(int)

                if not (
                    0 <= x < data.shape[0]
                    and 0 <= y < data.shape[1]
                    and 0 <= z < data.shape[2]
                ):
                    continue

                any_in_bounds = True
                lesion_id = int(data[x, y, z])

                if lesion_id > 0:
                    instance_mapping[str(lesion_id)] = int(row["label"])
                    matched_ids.add(lesion_id)
                    if candidate_idx == 0:
                        native_hits += 1
                    else:
                        flipped_xy_hits += 1
                    found = True
                    break

                # Local search around rounded coordinates to reduce rounding mismatch.
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            xx, yy, zz = x + dx, y + dy, z + dz
                            if (
                                0 <= xx < data.shape[0]
                                and 0 <= yy < data.shape[1]
                                and 0 <= zz < data.shape[2]
                            ):
                                nid = int(data[xx, yy, zz])
                                if nid > 0:
                                    instance_mapping[str(nid)] = int(row["label"])
                                    matched_ids.add(nid)
                                    local_search_hits += 1
                                    if candidate_idx == 0:
                                        native_hits += 1
                                    else:
                                        flipped_xy_hits += 1
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break

            if not found:
                if not any_in_bounds:
                    out_of_bounds += 1
                else:
                    background_hits += 1

        missing_ids = sorted(unique_id_set - matched_ids)

        log["mapped_ids_count"] = int(len(instance_mapping))
        log["missing_ids_count"] = int(len(missing_ids))
        log["out_of_bounds_count"] = int(out_of_bounds)
        log["background_hit_count"] = int(background_hits)
        log["native_hit_count"] = int(native_hits)
        log["flipped_xy_hit_count"] = int(flipped_xy_hits)
        log["local_search_hit_count"] = int(local_search_hits)
        log["matched_instance_ids"] = " ".join(map(str, sorted(matched_ids)))
        log["missing_instance_ids"] = " ".join(map(str, missing_ids))

        print(f"Processed case {case_id}: \n \t Mapped ids:{log.get('matched_instance_ids')} Missing ids: {log.get('missing_instance_ids')}")

        if len(case_rows) == 0:
            log["status"] = "no_csv_match"

        if not json_only:
            output_mask_path = output_dir / mask_path.name
            if output_mask_path.resolve() != mask_path.resolve():
                shutil.copy2(mask_path, output_mask_path)

        json_path = output_dir / f"{case_id}.json"
        with open(json_path, "w") as f:
            json.dump({"instances": instance_mapping}, f, indent=4)
        log["json_path"] = str(json_path)
    except Exception as e:
        log["status"] = "error"
        log["error_message"] = str(e)

    return log


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MedSAMv2 masks for nnDetection by creating JSON files with instance-to-class mappings."
    )
    parser.add_argument(
        "-c", "--csv", type=str, required=True, help="Path to the annotations CSV file."
    )
    parser.add_argument(
        "-m",
        "--masks",
        type=str,
        required=True,
        help="Path to the directory containing the masks.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Path to the directory to save the new masks.",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=4,
        help="Number of threads to use.",
    )
    parser.add_argument(
        "-j",
        "--json-only",
        action="store_true",
        help="Only create json files, do not copy masks to output directory",
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default=None,
        help="Path to save conversion log CSV. Defaults to <output-dir>/conversion_log.csv",
    )
    args = parser.parse_args()

    csv_path = args.csv
    mask_dir = Path(args.masks)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    print(f"Reading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    required_cols = ["SeriesInstanceUID", "CoordX", "CoordY", "CoordZ", "label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV.")

    df["sanitized_uid"] = (
        df["SeriesInstanceUID"].astype(str).str.replace(".", "_", regex=False)
    )

    mask_cases = discover_masks(mask_dir)
    print(f"Found {len(mask_cases)} mask files in {mask_dir}")
    if len(mask_cases) == 0:
        print("No masks found. Exiting.")
        return

    def process_wrapper(task):
        case_id, mask_path = task
        case_rows = df[df["sanitized_uid"] == case_id]
        return process_case_with_logging(
            case_id=case_id,
            mask_path=mask_path,
            case_rows=case_rows,
            output_dir=output_dir,
            json_only=args.json_only,
        )

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        logs = list(
            tqdm(
                executor.map(process_wrapper, mask_cases),
                total=len(mask_cases),
                desc="Processing masks",
            )
        )

    log_df = pd.DataFrame(logs)
    log_csv_path = (
        Path(args.log_csv) if args.log_csv else output_dir / "conversion_log.csv"
    )
    log_df.to_csv(log_csv_path, index=False)

    print(f"Saved conversion log to {log_csv_path}")
    print(
        f"success={(log_df['status'] == 'success').sum()} no_csv_match={(log_df['status'] == 'no_csv_match').sum()} error={(log_df['status'] == 'error').sum()}"
    )


if __name__ == "__main__":
    main()
