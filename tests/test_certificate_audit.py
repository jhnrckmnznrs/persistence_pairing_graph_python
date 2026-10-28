"""End-to-end smoke and reproducibility tests for the manifest certificate audit."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "certificate_audit.py"
spec = importlib.util.spec_from_file_location("certificate_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(audit)


def _manifest(path, volumes):
    path.write_text(json.dumps({
        "schema_version": 1, "volumes": volumes,
    }), encoding="utf-8")
    return path


def _run(manifest, output, *extra):
    return subprocess.run([
        sys.executable, str(SCRIPT), "run", "--manifest", str(manifest),
        "--output", str(output), "--radii", "0", "1", "2",
        "--max-events-per-dimension", "1",
        "--reference-max-grid-cells", "5000",
        "--max-reference-queries", "2",
        "--timeout-s", "60", *extra,
    ], capture_output=True, text=True, check=False, timeout=75)


def test_manifest_validation_and_deterministic_event_order(tmp_path):
    image = tmp_path / "data.npy"
    np.save(image, np.zeros((3, 3), dtype=np.uint8))
    manifest = _manifest(tmp_path / "inputs.json", [
        {"id": "first", "path": "data.npy", "group": "A",
         "voxel_spacing_mm": [0.025, 0.025]},
    ])
    rows = audit.validate_manifest(manifest)
    assert rows[0]["id"] == "first"
    assert rows[0]["path"] == str(image.resolve())
    assert rows[0]["voxel_spacing_mm"] == [0.025, 0.025]
    with pytest.raises(ValueError, match="Duplicate"):
        audit.validate_manifest(_manifest(manifest, [
            {"id": "first", "path": "data.npy"},
            {"id": "first", "path": "data.npy"},
        ]))
    with pytest.raises(ValueError, match="Voxel|voxel"):
        audit.validate_manifest(_manifest(manifest, [
            {"id": "first", "path": "data.npy",
             "voxel_spacing_mm": [0.0, 0.025]},
        ]))


def test_nested_region_boxes_and_candidate_enumeration():
    grid = (11, 11, 11)
    center = (5, 5, 5)
    boxes = [audit.region_bounds(center, grid, radius)
             for radius in (0, 1, 2, None)]
    for j in range(len(boxes)-1):
        assert all(boxes[j+1][0][a] <= boxes[j][0][a]
                   and boxes[j][1][a] <= boxes[j+1][1][a]
                   for a in range(3))
    image = np.zeros((5, 5, 5), dtype=np.uint8)
    image[2, 2, 2] = 90
    cs, n = audit.allowed_new_in_bounds(
        image, grid, 2, 90, *boxes[0],
        max_examined=100, max_new=100)
    assert len(cs) == 1 and n == 1
    with pytest.raises(MemoryError):
        audit.allowed_new_in_bounds(image, grid, 2, 90,
                                    *boxes[-1], max_examined=100,
                                    max_new=100)


def test_tiff_volume_end_to_end_and_full_reference(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    image = np.zeros((5, 5, 5), dtype=np.uint8)
    image[2, 2, 2] = 90
    tifffile.imwrite(tmp_path / "cavity.tif", image,
                     photometric="minisblack")
    path = _manifest(tmp_path / "manifest.json", [
        {"id": "cavity", "path": "cavity.tif",
         "voxel_spacing_mm": [0.02, 0.02, 0.025],
         "group": "synthetic", "split": "test"},
    ])
    result = _run(path, tmp_path / "audit", "--compare-component-reference")
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    folder = tmp_path / "audit"
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["events"] >= 1
    assert summary["regions"] >= 3
    assert summary["reference_comparisons"] >= 1
    assert summary["component_reference_comparisons"] >= 1
    assert summary["verified_reconstructions"] >= 1
    rows = [json.loads(x) for x in
            (folder / "cavity.jsonl").read_text().splitlines()]
    regions = [x for x in rows if x["kind"] == "region"]
    assert any(r.get("feasible") is True for r in regions)
    assert any(r.get("verified") is True for r in regions)
    assert all(r.get("status") == "ok" for r in regions)
    assert (folder / "events.csv").is_file()
    import csv
    with (folder / "events.csv").open(newline="", encoding="utf-8") as f:
        first_event = next(csv.DictReader(f))
    assert first_event["first_sufficient_tested_radius"] in ("0", "1", "2", "global")
    assert int(first_event["completed_region_queries"]) >= 3
    assert (folder / "regions.csv").is_file()
    vol = next(x for x in rows if x["kind"] == "volume")
    assert vol["input_sha256"] == audit.file_sha256(tmp_path / "cavity.tif")
    assert vol["voxel_spacing_mm"] == [0.02, 0.02, 0.025]


def test_npy_inputs_and_no_events_are_reported(tmp_path):
    image = np.zeros((2, 2, 2), dtype=np.uint8)
    np.save(tmp_path / "constant.npy", image)
    manifest = _manifest(tmp_path / "input.json",
                         [{"id": "constant", "path": "constant.npy"}])
    result = _run(manifest, tmp_path / "out", "--no-global")
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["events"] == 0
    assert summary["zero_event_volumes"] == ["constant"]
    inventory = summary["event_inventories"]["constant"]
    assert inventory.get("1", {}).get("finite_positive", 0) == 0
    assert inventory.get("2", {}).get("finite_positive", 0) == 0
    assert summary["status"] == "complete"
    records = [json.loads(x) for x in
               (tmp_path / "out/constant.jsonl").read_text().splitlines()]
    v = next(x for x in records if x["kind"] == "volume")
    assert v["filtration_distinct_levels"] == 1
    assert v["filtration_min"] == v["filtration_max"] == 0
    assert next(x for x in records if x["kind"] == "event_inventory")[
        "selection_status"] == "no_eligible_finite_positive_h1_h2"
    assert (tmp_path / "out/regions.csv").is_file()
    again = _run(manifest, tmp_path / "out", "--no-global")
    assert again.returncode != 0
    assert "not empty" in again.stderr
    over = _run(manifest, tmp_path / "out", "--overwrite", "--no-global")
    assert over.returncode == 0, over.stderr


def test_bad_digest_or_unavailable_resource_captured_as_failure(tmp_path):
    image = np.zeros((2, 2, 2), dtype=np.uint8)
    np.save(tmp_path / "case.npy", image)
    manifest = _manifest(tmp_path / "inputs.json", [{
        "id": "bad", "path": "case.npy", "sha256": "0" * 64,
    }])
    result = _run(manifest, tmp_path / "results")
    assert result.returncode == 1, result.stdout + "\n" + result.stderr
    info = json.loads((tmp_path / "results/summary.json").read_text())
    assert info["status"] == "failed"
    assert len(info["failed_workers"]) == 1
    rows = [json.loads(x) for x in
            (tmp_path / "results/bad.jsonl").read_text().splitlines()]
    assert any(x["kind"] == "failure" and x["error_type"] == "ValueError"
               for x in rows)


def test_explicit_bad_event_index_is_not_silently_ignored(tmp_path):
    np.save(tmp_path / "case.npy", np.zeros((3, 3, 3), dtype=np.uint8))
    manifest = _manifest(tmp_path / "inputs.json", [{
        "id": "bad_events", "path": "case.npy", "event_ids": [999999],
    }])
    result = _run(manifest, tmp_path / "audit")
    assert result.returncode == 1
    summary = json.loads((tmp_path / "audit/summary.json").read_text())
    assert summary["status"] == "failed"


def test_event_inventory_distinguishes_zero_persistence_and_essential_nodes():
    from persistence_pairing_graph._graph import PersistenceNode
    nodes = [
        PersistenceNode(1, 1, 2, 10, 10),
        PersistenceNode(1, 3, 4, 10, 12),
        PersistenceNode(2, 5, None, 12, float("inf")),
        PersistenceNode(0, 6, 7, 2, 7),
    ]
    inv = audit.count_persistence_events(nodes)
    assert inv["1"] == {
        "finite_positive": 1,
        "finite_zero": 1,
        "essential": 0,
        "max_finite_persistence": 2.0,
    }
    assert inv["2"]["essential"] == 1
    assert inv["0"]["finite_positive"] == 1


def test_uint16_low_codes_histogram_and_signed_offset_counts():
    low = np.array([1, 4, 1], dtype=np.uint16).reshape((1, 1, 3))
    q = audit.summarize_filtration(low)
    assert q["filtration_distinct_levels"] == 2
    assert q["code_below_32768"] == 3
    assert q["code_at_32768"] == 0
    assert q["code_above_32768"] == 0
    signed = np.array([32704, 32768, 32832],dtype=np.uint16).reshape((1,1,3))
    got = audit.summarize_filtration(signed)
    assert got["filtration_distinct_levels"] == 3
    assert (got["code_below_32768"], got["code_at_32768"],
            got["code_above_32768"]) == (1,1,1)
