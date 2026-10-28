"""DRP-372 Stage E/F/G read-only provenance preflight checks."""
from __future__ import annotations
import csv
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "inspect_drp372_reuse.py"
spec = importlib.util.spec_from_file_location("inspect_drp372_reuse", SCRIPT)
pre = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pre)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_stage_preflight_is_read_only_and_reports_mismatch(tmp_path):
    root = tmp_path / "drp372_256"
    write_csv(root / "sedt_stage_e" / "sedt_manifest.csv", [
        {"sample":"339_02","levels_per_voxel":"64","sedt_path":"old/foo.tif"},
        {"sample":"339_03","levels_per_voxel":"64","sedt_path":"old/bar.tif"},
    ])
    write_csv(root / "fpg_stage_f" / "fpg_batch_manifest.csv", [
        {"sample":"339_02","nodes_sha256":"aa","edges_sha256":"bb"},
        {"sample":"339_03","nodes_sha256":"cc","edges_sha256":"dd"},
    ])
    write_csv(root / "features_stage_g" / "combined_features.csv", [
        {"sample":"339_02","project_validation_group":"DRP339",
         "family_validation_group":"familyA","primary_outer_fold":"1","f":"5"},
        {"sample":"339_03","project_validation_group":"DRP339",
         "family_validation_group":"familyA","primary_outer_fold":"2","f":"7"},
    ])
    write_csv(root / "features_stage_g" / "stage_g_qc.csv", [
        {"sample":"339_02","node_sha256_observed":"aa","edge_sha256_observed":"bb"},
        {"sample":"339_03","node_sha256_observed":"cc","edge_sha256_observed":"dd"},
    ])
    (root / "features_stage_g" / "stage_g_summary.json").write_text(
        json.dumps({"levels_per_voxel":64.0,"zero_code":32768,
                    "successful_specimens":2,"status":"pass"}))
    (tmp_path / "drp372_prepare_sedt.py").write_text(
        "LEVELS_PER_VOXEL = 64\nZERO_CODE = 32768\n")
    report = pre.run_preflight(tmp_path)
    assert report["stage_g_summary"]["levels_per_voxel"] == 64.0
    assert report["stage_f_node_edge_hashes"]["node"]["matches"] == 2
    assert report["stage_f_node_edge_hashes"]["edge"]["matches"] == 2
    assert report["sample_set_differences"]["stage_e"]["stage_g_not_in_table"] == 0
    assert report["outer_fold_checks"]["project_validation_group"][
        "crossing_groups"]["DRP339"] == ["1", "2"]
    assert any("64" in x["source"]
               for x in report["source_sedt_script"]["matches"])
    assert len(report["warnings"]) >= 2
    assert not (root / "report.json").exists()


def test_missing_manifest_and_nonoverlapping_groups(tmp_path):
    root = tmp_path / "drp372_256"
    write_csv(root / "features_stage_g" / "combined_features.csv", [
        {"sample":"x","project_validation_group":"P","primary_outer_fold":"1"},
        {"sample":"y","project_validation_group":"P","primary_outer_fold":"1"},
    ])
    report = pre.run_preflight(tmp_path)
    assert not report["manifests"]["stage_e"]["exists"]
    assert report["outer_fold_checks"]["project_validation_group"]["crossing_groups"] == {}
    assert report["sample_set_differences"]["stage_f"]["stage_g_not_in_table"] == 2
