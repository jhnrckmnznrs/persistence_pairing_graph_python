"""Read-only DRP-372 provenance preflight; does not recompute any volume.

Run from the Python PPG repository:
  python benchmarks/inspect_drp372_reuse.py \
      --workspace ~/temp/drp372 \
      --output ~/temp/drp372/ppg_reuse_preflight.json

Inspects Stage E/F/G manifest schemas, the frozen 119-case sample sets,
and the original SEDT script for encoding constants. It never opens
256^3 volume data and never modifies the original research files.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re


FIELDS_OF_INTEREST = re.compile(
    r"sample|path|file|hash|sha|status|level|scale|offset|zero|dtype|sign|spacing|encoding",
    re.I,
)
SEDT_PATTERN = re.compile(
    r"levels_per_voxel|zero_code|32768|distance_transform_edt|"
    r"quantiz|scale|signed|uint16|(?:\b32\b)|(?:\b64\b)", re.I
)


def read_csv(path: Path):
    if not path.is_file():
        return {"exists": False, "path": str(path)}, []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    preview = [
        {k: str(v)[:160] for k, v in row.items()
         if FIELDS_OF_INTEREST.search(k)}
        for row in rows[:2]
    ]
    return {
        "exists": True, "path": str(path),
        "rows": len(rows), "columns": fieldnames,
        "preview": preview,
        "unique_samples": (len({r["sample"] for r in rows})
                           if "sample" in fieldnames else None),
    }, rows


def group_crossings(rows, group_field: str, fold_field="primary_outer_fold"):
    result: dict[str, set[str]] = {}
    for row in rows:
        group, fold = row.get(group_field), row.get(fold_field)
        if group and fold:
            result.setdefault(group, set()).add(fold)
    return {
        "group_count": len(result),
        "crossing_groups": {group: sorted(folds)
                            for group, folds in sorted(result.items())
                            if len(folds) > 1},
    }


def stage_g_hash_audit(qc, manifest):
    if not qc or not manifest:
        return {"status": "unavailable"}
    by_name = {r.get("sample"): r for r in manifest}
    out = {}
    for dimension in ("node", "edge"):
        observed = f"{dimension}_sha256_observed"
        expected = f"{dimension}s_sha256"
        valid = [r for r in qc if r.get("sample") in by_name
                 and r.get(observed) and by_name[r["sample"]].get(expected)]
        out[dimension] = {
            "checked": len(valid),
            "matches": sum(r[observed] == by_name[r["sample"]][expected]
                           for r in valid),
        }
    return out


def scan_sedt_script(path: Path):
    if not path.is_file():
        return {"exists": False, "path": str(path)}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    matching = [
        {"line": i + 1, "source": line.strip()[:200]}
        for i, line in enumerate(lines)
        if SEDT_PATTERN.search(line)
    ]
    return {
        "exists": True, "path": str(path),
        "matches": matching[:35], "total_matches": len(matching),
    }


def run_preflight(workspace: Path):
    workspace = workspace.expanduser().resolve()
    root = workspace / "drp372_256"
    files = {
        "stage_e": root / "sedt_stage_e" / "sedt_manifest.csv",
        "stage_f": root / "fpg_stage_f" / "fpg_batch_manifest.csv",
        "stage_g_features": root / "features_stage_g" / "combined_features.csv",
        "stage_g_metadata": root / "features_stage_g" / "cohort_metadata.csv",
        "stage_g_qc": root / "features_stage_g" / "stage_g_qc.csv",
        "cohort_v1_1": root / "cohort_qc_v1_1" / "complete_pairs.csv",
        "cohort_original": root / "cohort_qc" / "complete_pairs.csv",
    }
    records = {}
    tables = {}
    for name, path in files.items():
        records[name], tables[name] = read_csv(path)

    summary_path = root / "features_stage_g" / "stage_g_summary.json"
    stage_g = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    sample_sets = {name: {r["sample"] for r in table
                          if r.get("sample")}
                   for name, table in tables.items()}
    basis = sample_sets["stage_g_features"]
    matches = {
        name: {
            "stage_g_not_in_table": len(basis - ids),
            "table_not_in_stage_g": len(ids - basis),
        }
        for name, ids in sample_sets.items() if name != "stage_g_features"
    }
    metadata = tables["stage_g_features"]
    group_checks = {
        kind: group_crossings(metadata, kind)
        for kind in ("project_validation_group", "family_validation_group")
    }
    saved_scale = stage_g.get("levels_per_voxel")
    warnings = []
    if saved_scale is not None:
        warnings.append(
            f"Stage G reports levels_per_voxel={saved_scale}; "
            "verify the Stage E array encoding and script before "
            "using any newly generated SEDT."
        )
    if group_checks["project_validation_group"]["crossing_groups"]:
        warnings.append(
            "Some project groups cross primary_outer_fold boundaries; "
            "do not label folds project-disjoint without validating the "
            "original split policy."
        )
    if (records["stage_e"]["exists"] and
            records["stage_e"].get("rows") != len(basis)):
        warnings.append(
            "Stage E manifest row count differs from Stage G cohort; "
            "check selected specimen membership."
        )
    return {
        "workspace": str(workspace),
        "stage_g_summary": {
            k: stage_g.get(k)
            for k in ("status", "successful_specimens", "selected_specimens",
                      "levels_per_voxel", "zero_code", "script_version",
                      "stage_e_manifest_sha256", "stage_f_manifest_sha256")
        },
        "manifests": records,
        "sample_set_differences": matches,
        "stage_f_node_edge_hashes": stage_g_hash_audit(
            tables["stage_g_qc"], tables["stage_f"]),
        "outer_fold_checks": group_checks,
        "source_sedt_script": scan_sedt_script(
            workspace / "drp372_prepare_sedt.py"),
        "warnings": warnings,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    report = run_preflight(args.workspace)
    if args.output:
        args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output.expanduser().write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # Print a compact summary instead of hundreds of field listings.
    print("Stage G:", report["stage_g_summary"])
    for name, record in report["manifests"].items():
        print(name + ":", "rows=" + str(record.get("rows")),
              "present=" + str(record["exists"]))
        if name in ("stage_e", "cohort_v1_1", "cohort_original"):
            print("  columns:", record.get("columns", []))
            print("  examples:", record.get("preview", [])[:1])
    print("Stage F node/edge hashes:", report["stage_f_node_edge_hashes"])
    print("Fold crossing:", report["outer_fold_checks"])
    print("SEDT script lines:", report["source_sedt_script"].get("matches", [])[:24])
    for warning in report["warnings"]:
        print("WARNING:", warning)
    if args.output:
        print("Wrote:", args.output.expanduser())


if __name__ == "__main__":
    main()
