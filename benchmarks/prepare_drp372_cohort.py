"""Convert the frozen DRP-372 complete-case cohort into PPG pilot inputs.

The earlier DRP-372 cohort freeze has this layout:
  drp372_256/binary/<sample>_256.mat
  drp372_256/targets/<sample>_256_LBM.csv
  drp372_256/cohort_qc/complete_pairs.csv

The MATLAB-v7.3 HDF5 variable 'bin' is a 256^3 binary array with
0 = pore and 1 = solid. The primary historical filtration is the
full-volume signed Euclidean distance transform (SEDT), negative in pore
and positive in solid, encoded as uint16:
  q = clip(round(32768 + S*(d_solid - d_pore)), 0, 65535),
where S must be explicitly verified against the saved Stage E manifest.
Distance transforms are calculated on the FULL 256^3 original domain
before extracting native-resolution crops. The optional binary filtration
is a separate control. No resizing or interpolation is performed.

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
FILTRATIONS = ("sedt", "binary")


def signed_edt_uint16(binary: np.ndarray, *, levels_per_voxel: float) -> np.ndarray:
    """Reproduce the historical signed-distance uint16 filtration encoding.

    Negative in pore (bin=0), positive in solid (bin=1), unit voxel
    spacing. SciPy's default array-edge convention is retained.
    Always apply this to the FULL original domain before cropping;
    otherwise the transform develops artificial crop-boundary distances.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:
        raise RuntimeError("Signed EDT requires scipy; install '.[rock]'") from exc
    binary = np.asarray(binary)
    if binary.ndim != 3 or not np.isin(binary,[0,1]).all():
        raise ValueError("SEDT input must be a 3D binary pore/solid array")
    if not np.any(binary == 0) or not np.any(binary == 1):
        raise ValueError("SEDT requires both pore and solid phases")
    signed = distance_transform_edt(binary == 1)
    signed -= distance_transform_edt(binary == 0)
    if not math.isfinite(levels_per_voxel) or levels_per_voxel <= 0:
        raise ValueError("levels_per_voxel must be finite and positive")
    signed *= levels_per_voxel
    signed += 32768.0
    np.rint(signed,out=signed)
    np.clip(signed,0,65535,out=signed)
    return signed.astype(np.uint16)




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


def read_stage_e_manifest(path: Path) -> dict[str, dict]:
    """Index the frozen Stage E manifest by sample, never by stale absolute path."""
    if not path.is_file():
        raise FileNotFoundError(f"Saved Stage E manifest missing: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {
            "sample", "status", "input_binary_sha256_observed",
            "levels_per_voxel", "zero_code", "output_path", "output_sha256",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Saved Stage E manifest lacks required provenance columns")
        rows = {}
        for row in reader:
            sample = row["sample"].strip()
            if not SAMPLE_NAME.fullmatch(sample):
                raise ValueError(f"Invalid sample ID in Stage E: {sample}")
            if sample in rows:
                raise ValueError(f"Duplicate Stage E sample: {sample}")
            rows[sample] = row
    return rows


def load_saved_stage_e(root: Path, sample: str, row: dict, *,
                       levels_per_voxel: float, binary_sha256: str
                       ) -> tuple[np.ndarray, Path, str]:
    """Load the frozen full-domain TIFF, verifying encoded file and source hashes.

    Stage E absolute paths in the old manifest refer to ~/Desktop/drp372.
    Resolve by validated sample ID under the new root, never by using the
    obsolete recorded absolute location. Fail closed on any mismatch.
    """
    if row["status"] != "ok":
        raise ValueError(f"Stage E sample {sample} status is {row['status']!r}")
    if not math.isclose(float(row["levels_per_voxel"]), levels_per_voxel,
                        rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"Stage E levels_per_voxel differs from requested scale for {sample}")
    if int(row["zero_code"]) != 32768:
        raise ValueError(f"Unexpected Stage E zero_code for {sample}")
    if row["input_binary_sha256_observed"].lower() != binary_sha256.lower():
        raise ValueError(f"Stage E source binary SHA-256 mismatch for {sample}")
    basename = f"{sample}_sedt_u16.tif"
    if Path(row["output_path"]).name != basename:
        raise ValueError(f"Stage E output filename mismatch for {sample}")
    tif = root / "sedt_stage_e" / "sedt" / basename
    if not tif.is_file():
        raise FileNotFoundError(f"Saved Stage E TIFF missing: {tif}")
    digest = sha256(tif)
    if not row["output_sha256"] or digest.lower() != row["output_sha256"].lower():
        raise ValueError(f"Saved Stage E TIFF SHA-256 mismatch for {sample}")
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError("Stage E reuse needs tifffile; install '.[images]'") from exc
    sedt = tifffile.imread(tif)
    if sedt.shape != (256, 256, 256) or sedt.dtype != np.dtype("uint16"):
        raise ValueError(
            f"Stage E TIFF must be 256^3 uint16 for {sample}; "
            f"got {sedt.shape}, {sedt.dtype}")
    for field, actual in (("encoded_min", int(sedt.min())),
                          ("encoded_max", int(sedt.max()))):
        if row.get(field) and int(row[field]) != actual:
            raise ValueError(f"Stage E {field} mismatch for {sample}")
    return sedt, tif, digest


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
             filtrations: tuple[str, ...] = ("sedt", "binary"),
             levels_per_voxel: float | None = None,
             sedt_source: str = "stage-e",
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
    if not filtrations or any(f not in FILTRATIONS for f in filtrations):
        raise ValueError("Select binary and/or sedt filtrations")
    if sedt_source not in ("stage-e", "recompute"):
        raise ValueError("sedt_source must be stage-e or recompute")
    if "sedt" in filtrations and (levels_per_voxel is None or
            not math.isfinite(levels_per_voxel) or levels_per_voxel <= 0):
        raise ValueError("Specify --levels-per-voxel from the verified Stage E manifest; do not assume 32 or 64")
    if voxel_spacing_mm is not None and (
            len(voxel_spacing_mm) != 3 or
            any(v <= 0 or not math.isfinite(v) for v in voxel_spacing_mm)):
        raise ValueError("Voxel spacing must contain three positive finite values")
    rows = read_complete_cohort(cohort_csv)
    stage_e = (read_stage_e_manifest(root / "sedt_stage_e" / "sedt_manifest.csv")
               if "sedt" in filtrations and sedt_source == "stage-e" else {})
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
            full_bin = np.asarray(data[...],dtype=np.uint8)
            if not np.isin(full_bin,[0,1]).all():
                raise ValueError(f"Source geometry for {sample} contains values other than 0 and 1")
        # Prefer the original, hash-verified full-volume Stage E TIFF.
        # Only explicit sedt_source='recompute' performs SciPy EDT work.
        stage_row = stage_e.get(sample)
        source_sedt_path = None
        source_sedt_sha = None
        if "sedt" not in filtrations:
            full_sedt = None
        elif sedt_source == "stage-e":
            if stage_row is None:
                raise ValueError(f"Sample {sample} absent from frozen Stage E manifest")
            full_sedt, source_sedt_path, source_sedt_sha = load_saved_stage_e(
                root, sample, stage_row, levels_per_voxel=levels_per_voxel,
                binary_sha256=original_hash)
        else:
            full_sedt = signed_edt_uint16(
                full_bin, levels_per_voxel=levels_per_voxel)
        label = permeability_lus(row)
        for size in sorted(set(crop_sizes)):
            for position in origins:
                offset = crop_origin(tuple(full_bin.shape), size, position)
                slices = tuple(slice(i,i+size) for i in offset)
                binary_crop = np.ascontiguousarray(full_bin[slices],dtype=np.uint8)
                for filtration in filtrations:
                    cropped = (np.ascontiguousarray(full_sedt[slices],dtype=np.uint16)
                               if filtration == "sedt" else binary_crop)
                    if filtration == "sedt" and not np.array_equal(
                            cropped < 32768, binary_crop == 0):
                        raise ValueError(
                            f"SEDT phase/axis mismatch against source binary for "
                            f"{sample}, {position}, size={size}")
                    ident = f"{sample}__s{size}__{position}__{filtration}"
                    target = volumes_dir / f"{ident}.npy"
                    if target.exists() and not overwrite:
                        raise FileExistsError(target)
                    np.save(target,cropped,allow_pickle=False)
                    target_rel = target.relative_to(output).as_posix()
                    digest = sha256(target)
                    manifest["volumes"].append({
                        "id": ident,
                        "path": target_rel,
                        "group": (stage_row.get("family_validation_group")
                                  if stage_row is not None else None)
                                 or row.get("conservative_validation_group")
                                 or row.get("project_id")
                                 or sample.split("_")[0],
                        "split": "pilot",
                        "voxel_spacing_mm": list(voxel_spacing_mm)
                                            if voxel_spacing_mm is not None else None,
                        "sha256": digest,
                    })
                    metadata.append({
                        "id": ident, "sample": sample,
                        "project_id": row.get("project_id", ""),
                        "group": (stage_row.get("family_validation_group")
                                  if stage_row is not None else None)
                                 or row.get("conservative_validation_group", ""),
                        "family_validation_group": (
                            stage_row.get("family_validation_group", "")
                            if stage_row is not None else ""),
                        "project_validation_group": (
                            stage_row.get("project_validation_group", "")
                            if stage_row is not None else ""),
                        "historical_primary_outer_fold": (
                            stage_row.get("primary_outer_fold", "")
                            if stage_row is not None else ""),
                        "sedt_source": sedt_source if filtration == "sedt" else "binary",
                        "stage_e_original_tiff_path": (
                            str(source_sedt_path) if filtration == "sedt"
                            and source_sedt_path is not None else ""),
                        "stage_e_original_tiff_sha256": (
                            source_sedt_sha if filtration == "sedt"
                            and source_sedt_sha is not None else ""),
                        "original_mat_path": str(mat),
                        "original_mat_sha256": original_hash,
                        "cropped_npy_path": target_rel,
                        "cropped_npy_sha256": digest,
                        "source_shape": "256x256x256",
                        "crop_shape": f"{size}x{size}x{size}",
                        "origin_mode": position,
                        "origin_zyx": json.dumps(offset),
                        "pore_label": 0, "solid_label": 1,
                        "pore_fraction": round(float(np.mean(binary_crop == 0)), 8),
                        "permeability_lus": label,
                        "log10_permeability_lus": (
                            round(math.log10(label), 10) if label is not None else None),
                        "filtration": ( f"sedt_uint16_offset32768_scale{levels_per_voxel:g}_full256"
                                        if filtration == "sedt" else "binary_lower_star"),
                        "filtration_dtype": str(cropped.dtype),
                        "filtration_min": int(cropped.min()),
                        "filtration_max": int(cropped.max()),
                    })
        del full_sedt, full_bin
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
        "filtrations": list(filtrations),
        "sedt_quantization": (f"clip(rint(32768 + {levels_per_voxel:g}*(d_solid - d_pore)), 0, 65535)"
                              if "sedt" in filtrations else None),
        "levels_per_voxel": levels_per_voxel,
        "distance_transform_domain": "full_256_cube_before_crop",
        "sedt_source": sedt_source if "sedt" in filtrations else None,
        "stage_e_manifest": (str(root / "sedt_stage_e" / "sedt_manifest.csv")
                             if "sedt" in filtrations
                             and sedt_source == "stage-e" else None),
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
    parser.add_argument("--filtrations", nargs="+", choices=FILTRATIONS,
                        default=["sedt", "binary"])
    parser.add_argument("--voxel-spacing-mm", nargs=3, type=float)
    parser.add_argument("--levels-per-voxel", type=float,
                        help="Required with sedt; verify against Stage E manifest")
    parser.add_argument("--sedt-source", choices=("stage-e", "recompute"),
                        default="stage-e",
                        help="Reuse hash-verified Stage E TIFFs (default); recompute only explicitly")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = generate(
        cohort_csv=args.cohort_csv, root=args.root, output=args.output,
        sample_ids=args.sample, limit=args.limit,
        crop_sizes=tuple(args.sizes), origins=tuple(args.origins),
        filtrations=tuple(args.filtrations),
        levels_per_voxel=args.levels_per_voxel,
        sedt_source=args.sedt_source,
        voxel_spacing_mm=tuple(args.voxel_spacing_mm)
                         if args.voxel_spacing_mm else None,
        overwrite=args.overwrite)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
