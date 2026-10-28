"""Convert the frozen DRP-372 complete-case cohort into PPG pilot inputs.

The earlier DRP-372 cohort freeze has this layout:
  drp372_256/binary/<sample>_256.mat
  drp372_256/targets/<sample>_256_LBM.csv
  drp372_256/cohort_qc/complete_pairs.csv

The MATLAB-v7.3 HDF5 variable 'bin' is a 256^3 binary array with
0 = pore and 1 = solid. Native-resolution *crops* preserve the exact
original voxel values; there is NO resizing, interpolation or thresholding.

The output is a manifest compatible with certificate_audit.py plus an
independent CSV recording crop origins, porosities, raw file hashes and
the previously audited LBM permeability target (when unambiguous).

This script never downloads samples, copies source MAT files or alters
their phase convention. It does not claim full-volume topology invariance:
cropping changes global connectivity and may create boundary features.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


SAMPLE_NAME = re.compile(r"^[0-9]+_[0-9]+(?:_[0-9]+)?$")
ORIGINS = ("center", "quarter", "three_quarter")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def read_complete_cohort(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rdr = csv.DictReader(stream)
        mandatory = {"sample", "inclusion_status", "binary_present", "lbm_present"}
        if not mandatory.issubset(rdr.fieldnames or []):
            raise ValueError("Expected DRP-372 finalized complete_pairs.csv schema")
        rows = list(rdr)
    seen: set[str] = set()
    for row in rows:
        sample = row["sample"].strip()
        if not SAMPLE_NAME.fullmatch(sample):
            raise ValueError("Unexpected DRP-372 sample ID: " + sample)
        if sample in seen:
            raise ValueError("Duplicate sample " + sample)
        seen.add(sample)
        if row["inclusion_status"] != "included_complete_pair":
            raise ValueError("The supplied CSV contains excluded sample " + sample)
        for field in ("binary_present", "lbm_present"):
            if row[field].lower() not in ("true", "1"):
                raise ValueError("Incomplete binary/LBM pair for " + sample)
    return sorted(rows, key=lambda row: row["sample"])


def crop_origin(shape: tuple[int, ...], crop_size: int,
                position: str) -> tuple[int, ...]:
    if position not in ORIGINS:
        raise ValueError("Unknown crop position")
    if crop_size <= 0 or crop_size > min(shape):
        raise ValueError("Crop size exceeds source dimension")
    fraction = {"center": .5, "quarter": .25,
                "three_quarter": .75}[position]
    return tuple(max(0, min(n-crop_size,
                             int(round(n*fraction - crop_size/2))))
                 for n in shape)


def permeability_lus(row: dict) -> float | None:
    """Use only explicitly named source-unit permeability, never a proxy."""
    raw = row.get("permeability_values_json") or ""
    if not raw:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("permeability_values_json must be an object")
    matched = [v for k, v in value.items()
               if str(k).strip().casefold() == "permeability (lus)"]
    if len(matched) != 1:
        return None
    try:
        permeability = float(matched[0])
    except (ValueError, TypeError):
        return None
    return permeability if math.isfinite(permeability) and permeability > 0 else None


def generate(*, cohort_csv: Path, root: Path, output: Path,
             sample_ids: list[str] | None = None,
             limit: int | None = 5,
             crop_sizes: tuple[int, ...] = (12, 24),
             origins: tuple[str, ...] = ("center", "quarter"),
             voxel_spacing_mm: tuple[float, ...] | None = None,
             overwrite: bool = False) -> dict:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Install h5py to read the DRP-372 MATLAB/HDF5 files") from exc
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if any(s < 2 for s in crop_sizes):
        raise ValueError("Crop dimensions must be >=2")
    if any(o not in ORIGINS for o in origins):
        raise ValueError("Unsupported crop origin")
    if voxel_spacing_mm is not None and (
            len(voxel_spacing_mm) != 3 or
            any(v <= 0 or not math.isfinite(v) for v in voxel_spacing_mm)):
        raise ValueError("Voxel spacing must contain three positive finite values")
    rows = read_complete_cohort(cohort_csv)
    ids = set(sample_ids or [])
    if ids:
        available = {row["sample"] for row in rows}
        unknown = ids - available
        if unknown:
            raise ValueError("Requested samples absent from complete cohort: " +
                             ",".join(sorted(unknown)))
        rows = [row for row in rows if row["sample"] in ids]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No DRP-372 sample chosen")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError("Output directory is nonempty; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    volumes_dir = output / "volumes"
    volumes_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"schema_version": 1, "volumes": []}
    metadata: list[dict] = []
    for row in rows:
        sample = row["sample"]
        mat = root / "binary" / f"{sample}_256.mat"
        if not mat.is_file():
            raise FileNotFoundError(f"DRP-372 geometry missing: {mat}")
        if row.get("binary_sha256") and (
                sha256(mat).lower() != row["binary_sha256"].lower()):
            raise ValueError(f"Frozen geometry checksum mismatch: {sample}")
        original_hash = sha256(mat)
        with h5py.File(mat, "r") as h:
            if "bin" not in h:
                raise ValueError(f"Missing MATLAB variable bin in {sample}")
            data = h["bin"]
            if len(data.shape) != 3 or any(int(x) != 256 for x in data.shape):
                raise ValueError(f"Expected 256^3 HDF5 bin array for {sample}; got {data.shape}")
            if np.dtype(data.dtype).kind not in "biu":
                raise ValueError(f"Expected numeric binary data for {sample}")
            for size in sorted(set(crop_sizes)):
                for position in origins:
                    offset = crop_origin(tuple(data.shape), size, position)
                    sub = np.asarray(data[tuple(slice(i, i+size) for i in offset)])
                    # Never silently reinterpret phase or binarize nonbinary data.
                    if not np.isin(sub, [0, 1]).all():
                        raise ValueError(f"DRP-372 nonbinary values in {sample} crop {position}")
                    pore = np.ascontiguousarray(sub, dtype=np.uint8)
                    ident = f"{sample}__s{size}__{position}"
                    target = volumes_dir / f"{ident}.npy"
                    if target.exists() and not overwrite:
                        raise FileExistsError(target)
                    np.save(target, pore, allow_pickle=False)
                    target_rel = target.relative_to(output).as_posix()
                    label = permeability_lus(row)
                    manifest["volumes"].append({
                        "id": ident,
                        "path": target_rel,
                        "group": row.get("conservative_validation_group")
                                 or row.get("project_id")
                                 or sample.split("_")[0],
                        "split": "pilot",
                        "voxel_spacing_mm": list(voxel_spacing_mm)
                                            if voxel_spacing_mm is not None else None,
                        "sha256": sha256(target),
                    })
                    metadata.append({
                        "id": ident, "sample": sample,
                        "project_id": row.get("project_id", ""),
                        "group": row.get("conservative_validation_group", ""),
                        "original_mat_path": str(mat),
                        "original_mat_sha256": original_hash,
                        "cropped_npy_path": target_rel,
                        "cropped_npy_sha256": sha256(target),
                        "source_shape": "256x256x256",
                        "crop_shape": f"{size}x{size}x{size}",
                        "origin_mode": position,
                        "origin_zyx": json.dumps(offset),
                        "pore_label": 0, "solid_label": 1,
                        "pore_fraction": round(float(np.mean(pore == 0)), 8),
                        "permeability_lus": label,
                        "log10_permeability_lus": (
                            round(math.log10(label), 10) if label is not None else None),
                        "filtration": "binary_lower_star",
                    })
    (output / "cohort_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    fields = list(metadata[0])
    with (output / "crop_provenance.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metadata)
    summary = {
        "source": "DRP-372 frozen complete_pairs.csv",
        "geometry_variable": "bin", "pore_label": 0, "solid_label": 1,
        "filtration": "binary_lower_star",
        "samples": len(rows), "crops": len(metadata),
        "crop_sizes": sorted(set(crop_sizes)), "origins": list(origins),
        "permeability_with_valid_source_unit": sum(
            x["permeability_lus"] is not None for x in metadata),
        "warning": "These are cropped local complexes; not full-volume PPGs or permeability measurements.",
    }
    (output / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-csv", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--sizes", nargs="+", type=int, default=[12, 24])
    parser.add_argument("--origins", nargs="+", choices=ORIGINS,
                        default=["center", "quarter"])
    parser.add_argument("--voxel-spacing-mm", nargs=3, type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = generate(
        cohort_csv=args.cohort_csv, root=args.root, output=args.output,
        sample_ids=args.sample, limit=args.limit,
        crop_sizes=tuple(args.sizes), origins=tuple(args.origins),
        voxel_spacing_mm=tuple(args.voxel_spacing_mm)
                         if args.voxel_spacing_mm else None,
        overwrite=args.overwrite)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
