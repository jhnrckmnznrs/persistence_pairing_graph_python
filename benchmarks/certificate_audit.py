"""Manifest-driven, resource-bounded PPG certificate and localization audit.

Usage:
  python benchmarks/certificate_audit.py run --manifest cohort.json --output audit/
  python benchmarks/certificate_audit.py run --manifest cohort.json --output audit/ --radii 0 1 2 4

Each volume runs in a separate subprocess. No input arrays or certificate
chains are copied into the output, and no data are uploaded anywhere.

A region restricts only the NEW (q+1)-cells at the selected event.
All old corrections remain unrestricted, as required by relative homology.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import itertools
import json
import os
from pathlib import Path
import platform
try:
    import resource
except ImportError:  # Windows has no POSIX resource module
    resource = None
import subprocess
import sys
from time import perf_counter
import traceback
from typing import Any

import numpy as np


SCHEMA_VERSION = 1


def validate_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or obj.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Manifest requires schema_version=1")
    entries = obj.get("volumes")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest requires a nonempty volumes array")
    used = set()
    result = []
    for value in entries:
        if not isinstance(value, dict):
            raise ValueError("Each volume must be an object")
        identifier = value.get("id")
        if (not isinstance(identifier, str) or not identifier or
                not all(c.isalnum() or c in "-_" for c in identifier)):
            raise ValueError("Volume id must use only letters, numbers, - or _")
        if identifier in used:
            raise ValueError("Duplicate volume id: " + identifier)
        used.add(identifier)
        p = value.get("path")
        if not isinstance(p, str) or not p:
            raise ValueError("Each volume needs a path")
        resolved = (manifest_path.parent / p).resolve()
        if resolved.suffix.lower() not in (".tif", ".tiff", ".npy"):
            raise ValueError("Supported input files are TIFF or NPY")
        if not resolved.is_file():
            raise FileNotFoundError(str(resolved))
        spacing = value.get("voxel_spacing_mm")
        if spacing is not None and (
            not isinstance(spacing, list) or len(spacing) not in (2, 3)
            or any(not isinstance(x, (float, int)) or x <= 0
                   or not np.isfinite(x) for x in spacing)):
            raise ValueError("voxel_spacing_mm must contain 2 or 3 positive numbers")
        digest = value.get("sha256")
        if digest is not None and (
            not isinstance(digest, str) or len(digest) != 64 or
            any(c not in "0123456789abcdefABCDEF" for c in digest)):
            raise ValueError("sha256 must be a 64-digit hexadecimal digest")
        events = value.get("event_ids")
        if events is not None and (
            not isinstance(events, list) or any(
                not isinstance(x, int) or isinstance(x, bool) or x < 0
                for x in events) or len(set(events)) != len(events)):
            raise ValueError("event_ids must be distinct nonnegative integers")
        result.append({
            "id": identifier, "path": str(resolved),
            "group": value.get("group"), "split": value.get("split"),
            "voxel_spacing_mm": spacing, "expected_sha256": digest,
            "event_ids": events,
        })
    return result


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_scalar_volume(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        data = np.load(path, allow_pickle=False, mmap_mode='r')
    else:
        try:
            import tifffile
        except ImportError as exc:
            raise RuntimeError("TIFF reading requires pip install '.[images]'") from exc
        data = tifffile.imread(path)
    if data.ndim not in (2, 3) or data.dtype not in (
        np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError("Input must be a single 2D/3D uint8 or uint16 scalar volume")
    if not all(data.shape):
        raise ValueError("Empty scalar dimension")
    return np.asarray(data)


def select_events(nodes, *, max_per_dimension: int, event_ids=None,
                  dimensions=(1, 2)) -> list[int]:
    """Stable ranking: dimension, descending persistence, event, birth ID."""
    if event_ids is not None:
        for i in event_ids:
            if i >= len(nodes):
                raise IndexError("Explicit event index exceeds node count")
            n = nodes[i]
            if n.essential or n.persistence <= 0 or n.dimension not in dimensions:
                raise ValueError("Explicit event is not an eligible finite positive-length class")
        return list(event_ids)
    chosen = []
    for q in dimensions:
        available = [i for i, node in enumerate(nodes)
                     if node.dimension == q and not node.essential
                     and node.persistence > 0]
        available.sort(key=lambda i: (-nodes[i].persistence,
                                      nodes[i].death_value,
                                      nodes[i].birth_cell, i))
        chosen.extend(available[:max_per_dimension])
    return chosen


def count_persistence_events(nodes) -> dict[str, dict]:
    """Report all dimensions before selecting finite positive H1/H2 events.

    Zero-persistence intervals are counted separately; they are not
    eligible death-event certificates under the current protocol.
    """
    counts: dict[str, dict] = {}
    for node in nodes:
        q = str(int(node.dimension))
        result = counts.setdefault(q, {
            "finite_positive": 0,
            "finite_zero": 0,
            "essential": 0,
            "max_finite_persistence": 0,
        })
        if node.essential:
            result["essential"] += 1
        elif node.persistence > 0:
            result["finite_positive"] += 1
            result["max_finite_persistence"] = max(
                result["max_finite_persistence"], float(node.persistence))
        else:
            result["finite_zero"] += 1
    return counts


def summarize_filtration(image: np.ndarray) -> dict:
    histogram = np.bincount(np.asarray(image).ravel())
    result = {
        "filtration_min": int(image.min()),
        "filtration_max": int(image.max()),
        "filtration_distinct_levels": int(np.count_nonzero(histogram)),
    }
    if image.dtype == np.dtype("uint16"):
        # These are only intensity-code bins. Their interpretation as
        # pore/solid is specific to DRP-372's offset-32768 SEDT.
        result["code_below_32768"] = int(histogram[:32768].sum())
        result["code_at_32768"] = int(histogram[32768])
        result["code_above_32768"] = int(histogram[32769:].sum())
    return result


def region_bounds(center: tuple[int, ...], grid_shape: tuple[int, ...],
                  radius: int | None):
    if radius is None:
        return tuple(0 for _ in grid_shape), tuple(n - 1 for n in grid_shape)
    if radius < 0:
        raise ValueError("Radii must be nonnegative")
    # Each input pixel/voxel occupies two doubled-grid coordinate steps.
    return (tuple(max(0, c - 2 * radius) for c in center),
            tuple(min(n - 1, c + 2 * radius)
                  for c, n in zip(center, grid_shape)))


def allowed_new_in_bounds(image, grid_shape, q, t, lower, upper, *,
                          max_examined: int, max_new: int):
    """Generate candidate cells without enumerating all old sublevel cells."""
    from persistence_pairing_graph._transition_witness import _original_cell_birth
    count = 1
    for a, b in zip(lower, upper):
        count *= b - a + 1
    if count > max_examined:
        raise MemoryError("Region exceeds max_examined_region_cells")
    new = []
    for coords in itertools.product(
            *(range(a, b + 1) for a, b in zip(lower, upper))):
        if sum(c & 1 for c in coords) != q + 1:
            continue
        cell = int(np.ravel_multi_index(coords, grid_shape))
        if _original_cell_birth(image, cell, grid_shape) == t:
            new.append(cell)
            if len(new) > max_new:
                raise MemoryError("Region exceeds max_new_cells")
    return new, count


def rss_kib() -> int | None:
    # ru_maxrss uses KiB on Linux; other platforms may use different units.
    if sys.platform != "linux" or resource is None:
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def emit(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def verify_reconstructed(image, grid_shape, witness, certificate, allowed) -> dict:
    from persistence_pairing_graph._lift import boundary_old_chain
    from persistence_pairing_graph._transition_witness import _original_cell_birth

    old = list(map(int, certificate.old_filling_cells))
    new = list(map(int, certificate.new_filling_cells))
    if not certificate.feasible or not certificate.old_filling_reconstructed:
        raise AssertionError("Expected a reconstructed feasible certificate")
    if not set(new).issubset(allowed):
        raise AssertionError("New filling escaped the permitted region")
    if any(_original_cell_birth(image, c, grid_shape) >= witness.event_value
           for c in old):
        raise AssertionError("Old correction escaped the pre-event filtration")
    result = boundary_old_chain([*old, *new], grid_shape)
    if not np.array_equal(result, witness.relation_support):
        raise AssertionError("Exact original-grid boundary mismatch")
    digest = hashlib.sha256(
        np.asarray(sorted([*old, *new]), dtype="<i8").tobytes()
    ).hexdigest()
    return {"verified": True, "old_cells": len(old),
            "new_cells": len(new), "filling_sha256": digest}


def benchmark_volume(entry, opts, out_jsonl: Path) -> None:
    """Process one volume. Failures are written before being re-raised."""
    from persistence_pairing_graph import (
        PPGAnalysis, SparseOldLevelOracle, certify_relative_filling,
    )

    path = Path(entry["path"])
    actual_sha = file_sha256(path)
    if entry.get("expected_sha256") and (
            entry["expected_sha256"].lower() != actual_sha):
        raise ValueError("Input SHA-256 does not match manifest")
    t0 = perf_counter()
    image = load_scalar_volume(path)
    if entry.get("voxel_spacing_mm") is not None and (
            len(entry["voxel_spacing_mm"]) != image.ndim):
        raise ValueError("Voxel spacing dimensionality differs from image")
    grid_shape = tuple(2 * n + 1 for n in image.shape)
    volume_info = {
        "kind": "volume", "id": entry["id"], "input_sha256": actual_sha,
        "shape": list(map(int, image.shape)), "dtype": str(image.dtype),
        "grid_cells": int(np.prod(grid_shape)),
        "group": entry.get("group"), "split": entry.get("split"),
        "voxel_spacing_mm": entry.get("voxel_spacing_mm"),
        "load_and_hash_s": round(perf_counter()-t0, 6),
        **summarize_filtration(image),
    }
    if np.prod(grid_shape) > opts["max_original_cells"]:
        raise MemoryError("Input doubled grid exceeds max_original_cells")
    t0 = perf_counter()
    with PPGAnalysis(
        image, retain_inclusion_trace=True,
        max_original_cells=opts["max_original_cells"],
        max_chain_terms=opts["max_chain_terms"],
        max_work=opts["max_work"],
        max_basis_columns=opts["max_basis_columns"],
        max_basis_entries=opts["max_basis_entries"],
    ) as analysis:
        volume_info["ppg_build_s"] = round(perf_counter()-t0, 6)
        volume_info["nodes"] = len(analysis.nodes)
        volume_info["compressed_generators"] = len(analysis.compressed)
        volume_info["canceled_pairs"] = int(analysis.compressed.lift_trace.n_pairs)
        volume_info["peak_rss_kib_after_build"] = rss_kib()
        inventory = count_persistence_events(analysis.nodes)
        volume_info["persistence_event_inventory"] = inventory
        emit(out_jsonl, volume_info)

        selected = select_events(
            analysis.nodes,
            max_per_dimension=opts["max_events_per_dimension"],
            event_ids=entry.get("event_ids"),
        )
        emit(out_jsonl, {
            "kind": "event_inventory", "id": entry["id"],
            "by_dimension": inventory,
            "eligible_h1": inventory.get("1", {}).get("finite_positive", 0),
            "eligible_h2": inventory.get("2", {}).get("finite_positive", 0),
            "selected_events": selected,
            "selection_status": ("selected" if selected
                                 else "no_eligible_finite_positive_h1_h2"),
        })
        if not selected:
            # Avoid needless oracle indexing and report why no queries ran.
            return
        start = perf_counter()
        oracle = SparseOldLevelOracle(
            analysis, max_output_cells=opts["max_chain_terms"],
            max_preimage_work=opts["max_work"],
        )
        emit(out_jsonl, {
            "kind": "oracle", "id": entry["id"],
            "init_s": round(perf_counter()-start, 6),
            "selected_events": selected, "peak_rss_kib": rss_kib(),
        })
        reference_queries = 0

        for node_id in selected:
            node = analysis.nodes[node_id]
            event = {
                "kind": "event", "id": entry["id"], "event_id": int(node_id),
                "dimension": node.dimension, "birth_value": int(node.birth_value),
                "death_value": int(node.death_value),
                "persistence": int(node.death_value-node.birth_value),
            }
            try:
                start = perf_counter()
                witness = analysis.witness(node_id, solver="replay")
                event["witness_s"] = round(perf_counter()-start, 6)
                event["relation_cells"] = len(witness.relation_support)
                event["witness_filling_cells"] = len(witness.filling_grid_indices)
                event["targets"] = len(witness.target_nodes)
                death_cell = int(analysis.compressed.original_grid_indices[node.death_cell])
                center = tuple(map(int, np.unravel_index(death_cell, grid_shape)))
                event["critical_cell_center"] = center

                if opts["measure_closure"]:
                    start = perf_counter()
                    closure = analysis.replay_dependency_closure(node_id)
                    event["closure_s"] = round(perf_counter()-start, 6)
                    event["closure_reachable"] = closure.reachable_records
                    event["closure_replayed"] = closure.replayed_records
                    event["closure_arcs"] = closure.dependency_arcs
                    event["closure_depth"] = closure.maximum_observed_depth
                    event["closure_fraction"] = closure.reachable_fraction
                event["status"] = "ok"
                event["peak_rss_kib"] = rss_kib()
                emit(out_jsonl, event)
            except (ValueError, MemoryError, RuntimeError, IndexError) as exc:
                event.update(status="error", error_type=type(exc).__name__,
                             error=str(exc))
                emit(out_jsonl, event)
                continue

            previous_feasible = False
            reconstructed = False
            radii = list(opts["radii"]) + ([None] if opts["include_global"] else [])
            for radius in radii:
                lo, hi = region_bounds(center, grid_shape, radius)
                label = "global" if radius is None else str(radius)
                result = {
                    "kind": "region", "id": entry["id"], "event_id": int(node_id),
                    "dimension": node.dimension, "radius_voxels": label,
                    "lower": lo, "upper": hi, "status": "ok",
                    "reference_status": "not_requested",
                }
                start = perf_counter()
                try:
                    new, examined = allowed_new_in_bounds(
                        image, grid_shape, node.dimension, node.death_value,
                        lo, hi, max_examined=opts["max_examined_region_cells"],
                        max_new=opts["max_new_cells"],
                    )
                    result["candidate_new_cells"] = len(new)
                    result["examined_grid_cells"] = examined
                    result["candidate_enumeration_s"] = round(perf_counter()-start, 6)

                    elapsed = []
                    decisions = []
                    certificate = None
                    for _ in range(opts["decision_repeats"]):
                        start_decision = perf_counter()
                        certificate = oracle.restricted_relative(
                            witness.relation_support, new,
                            dimension=node.dimension, event_value=node.death_value,
                            reconstruct=False,
                            max_new_cells=opts["max_new_cells"],
                            max_relative_rows=opts["max_relative_rows"],
                            max_relative_work=opts["max_relative_work"],
                        )
                        elapsed.append(perf_counter()-start_decision)
                        decisions.append(bool(certificate.feasible))
                    if len(set(decisions)) != 1:
                        raise AssertionError("Repeated feasibility decisions disagree")
                    feasible = decisions[0]
                    if previous_feasible and not feasible:
                        raise AssertionError("Feasibility violates nested-region monotonicity")
                    previous_feasible |= feasible
                    result["feasible"] = feasible
                    result["decision_seconds"] = [round(s, 6) for s in elapsed]
                    result["relative_kernel_generators"] = (
                        certificate.relative_kernel_generators)
                    result["coordinate_rank"] = certificate.coordinate_rank
                    if not feasible:
                        result["dual_coordinates"] = len(certificate.obstruction_coordinates)
                        start_verify = perf_counter()
                        valid_dual = oracle.evaluate_cocycle(
                            certificate.obstruction_coordinates,
                            witness.relation_support,
                            dimension=node.dimension, event_value=node.death_value,
                        ) == 1
                        if not valid_dual:
                            raise AssertionError("Dual obstruction fails pairing with relation")
                        result["dual_verified_on_relation"] = True
                        result["dual_check_s"] = round(perf_counter()-start_verify, 6)

                    if feasible and not reconstructed and opts["reconstruct_first"]:
                        start_reconstruct = perf_counter()
                        full = oracle.restricted_relative(
                            witness.relation_support, new,
                            dimension=node.dimension, event_value=node.death_value,
                            reconstruct=True,
                            max_new_cells=opts["max_new_cells"],
                            max_relative_rows=opts["max_relative_rows"],
                            max_relative_work=opts["max_relative_work"],
                        )
                        result["reconstruction_s"] = round(
                            perf_counter()-start_reconstruct, 6)
                        result.update(verify_reconstructed(
                            image, grid_shape, witness, full, set(new)))
                        reconstructed = True

                    if (np.prod(grid_shape) <= opts["reference_max_grid_cells"]
                            and reference_queries < opts["max_reference_queries"]):
                        start_reference = perf_counter()
                        reference = certify_relative_filling(
                            image, witness.relation_support, new,
                            dimension=node.dimension,
                            event_value=node.death_value, old_solver="full",
                            max_grid_cells=opts["reference_max_grid_cells"],
                            max_old_cells=opts["reference_max_old_cells"],
                            max_boundary_rows=opts["reference_max_boundary_rows"],
                        )
                        result["reference_s"] = round(
                            perf_counter()-start_reference, 6)
                        result["reference_old_columns"] = reference.old_candidate_count
                        result["reference_status"] = "verified"
                        if bool(reference.feasible) != feasible:
                            raise AssertionError("Sparse vs full reference feasibility mismatch")
                        if opts["compare_component_reference"]:
                            start_component = perf_counter()
                            component = certify_relative_filling(
                                image, witness.relation_support, new,
                                dimension=node.dimension,
                                event_value=node.death_value, old_solver="component",
                                max_grid_cells=opts["reference_max_grid_cells"],
                                max_old_cells=opts["reference_max_old_cells"],
                                max_boundary_rows=opts["reference_max_boundary_rows"],
                            )
                            result["component_reference_s"] = round(
                                perf_counter()-start_component, 6)
                            result["component_old_columns"] = component.old_candidate_count
                            if bool(component.feasible) != feasible:
                                raise AssertionError(
                                    "Sparse vs incidence-component feasibility mismatch")
                            result["component_reference_status"] = "verified"
                        else:
                            result["component_reference_status"] = "not_requested"
                        reference_queries += 1
                    else:
                        result["reference_status"] = "skipped_resource_budget"
                        result["component_reference_status"] = "skipped_resource_budget"
                    result["peak_rss_kib"] = rss_kib()
                except (ValueError, MemoryError, RuntimeError, AssertionError) as exc:
                    result.update(status="error", error_type=type(exc).__name__,
                                  error=str(exc), peak_rss_kib=rss_kib())
                emit(out_jsonl, result)


def _worker(config_path: Path, result_path: Path) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    entry, opts = config["entry"], config["options"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        result_path.unlink()
    try:
        benchmark_volume(entry, opts, result_path)
        return 0
    except BaseException as exc:
        emit(result_path, {
            "kind": "failure", "id": entry["id"],
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(limit=5),
            "peak_rss_kib": rss_kib(),
        })
        return 1


def _csv(path: Path, rows: list[dict]) -> None:
    columns = sorted(set().union(*(r.keys() for r in rows))) if rows else ["id", "status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (json.dumps(value) if isinstance(
                value, (tuple, list, dict)) else value)
                for key, value in row.items()})


def software_revision() -> dict:
    try:
        package_version = version("persistence-pairing-graph")
    except PackageNotFoundError:
        package_version = None
    try:
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=3, check=False)
        revision = git.stdout.strip() if git.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        revision = None
    return {"package_version": package_version, "git_commit": revision}


def run(args) -> int:
    entries = validate_manifest(args.manifest.resolve())
    if args.max_events_per_dimension < 1 or args.decision_repeats < 1:
        raise ValueError("Counts and repeats must be positive")
    if any(x < 0 for x in args.radii):
        raise ValueError("Radii must be nonnegative")
    if args.timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError("Output directory is not empty (use --overwrite)")
    output.mkdir(parents=True, exist_ok=True)
    options = {
        "radii": sorted(set(args.radii)),
        "include_global": not args.no_global,
        "max_events_per_dimension": args.max_events_per_dimension,
        "decision_repeats": args.decision_repeats,
        "reconstruct_first": not args.no_reconstruction,
        "measure_closure": not args.no_closure,
        "max_original_cells": args.max_original_cells,
        "max_chain_terms": args.max_chain_terms,
        "max_work": args.max_work,
        "max_basis_columns": args.max_basis_columns,
        "max_basis_entries": args.max_basis_entries,
        "max_new_cells": args.max_new_cells,
        "max_relative_rows": args.max_relative_rows,
        "max_relative_work": args.max_relative_work,
        "max_examined_region_cells": args.max_examined_region_cells,
        "reference_max_grid_cells": args.reference_max_grid_cells,
        "reference_max_old_cells": args.reference_max_old_cells,
        "reference_max_boundary_rows": args.reference_max_boundary_rows,
        "max_reference_queries": args.max_reference_queries,
        "compare_component_reference": args.compare_component_reference,
    }
    (output / "configuration.json").write_text(json.dumps({
        "manifest": entries, "options": options,
        "environment": {"python": sys.version, "numpy": np.__version__,
                        "platform": platform.platform(),
                        "software": software_revision()},
    }, indent=2) + "\n", encoding="utf-8")
    failures = []
    for entry in entries:
        config_path = output / (entry["id"] + "_worker_config.json")
        result_path = output / (entry["id"] + ".jsonl")
        config_path.write_text(json.dumps({
            "entry": entry, "options": options,
        }) + "\n", encoding="utf-8")
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "_worker", "--config", str(config_path),
               "--result", str(result_path)]
        try:
            run_result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=args.timeout_s,
                check=False)
            if run_result.returncode != 0:
                failures.append({"id": entry["id"], "status": "worker_failure",
                                 "returncode": run_result.returncode,
                                 "stderr_tail": run_result.stderr[-2000:]})
        except subprocess.TimeoutExpired:
            failures.append({"id": entry["id"], "status": "timeout",
                             "timeout_s": args.timeout_s})
    records = []
    for entry in entries:
        path = output / (entry["id"] + ".jsonl")
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
    event_rows = [x for x in records if x.get("kind") == "event"]
    region_rows = [x for x in records if x.get("kind") == "region"]
    inventory_rows = [x for x in records if x.get("kind") == "event_inventory"]
    for event in event_rows:
        related = [r for r in region_rows
                   if r.get("id") == event["id"]
                   and r.get("event_id") == event["event_id"]]
        event["completed_region_queries"] = sum(
            r.get("status") == "ok" for r in related)
        event["failed_region_queries"] = sum(
            r.get("status") != "ok" for r in related)
        first = next((r for r in related
                      if r.get("status") == "ok" and r.get("feasible") is True),
                     None)
        event["first_sufficient_tested_radius"] = (
            first["radius_voxels"] if first is not None else None)
        event["regions_exhaustively_decided"] = bool(related) and all(
            r.get("status") == "ok" for r in related)
    _csv(output / "events.csv", event_rows)
    _csv(output / "regions.csv", region_rows)
    issues = [
        {"id": r.get("id"), "event_id": r.get("event_id"),
         "radius": r.get("radius_voxels"), "error": r.get("error")}
        for r in records if r.get("kind") == "failure" or r.get("status") == "error"]
    summary = {
        "schema_version": SCHEMA_VERSION, "volumes": len(entries),
        "events": len(event_rows), "regions": len(region_rows),
        "zero_event_volumes": [
            x["id"] for x in inventory_rows
            if x["selection_status"] == "no_eligible_finite_positive_h1_h2"
        ],
        "event_inventories": {
            x["id"]: x["by_dimension"] for x in inventory_rows
        },
        "feasible_regions": sum(r.get("feasible") is True for r in region_rows),
        "infeasible_regions": sum(r.get("feasible") is False for r in region_rows),
        "verified_reconstructions": sum(r.get("verified") is True for r in region_rows),
        "reference_comparisons": sum(
            r.get("reference_status") == "verified" for r in region_rows),
        "component_reference_comparisons": sum(
            r.get("component_reference_status") == "verified" for r in region_rows),
        "failed_workers": failures, "recorded_errors": issues,
        "status": "failed" if failures or issues else "complete",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if failures or issues else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    worker = subs.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--result", type=Path, required=True)
    command = subs.add_parser("run", help="Run manifest-driven isolated-volume audit")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--overwrite", action="store_true")
    command.add_argument("--radii", nargs="+", type=int, default=[0, 1, 2, 4])
    command.add_argument("--no-global", action="store_true")
    command.add_argument("--max-events-per-dimension", type=int, default=2)
    command.add_argument("--decision-repeats", type=int, default=2)
    command.add_argument("--no-reconstruction", action="store_true")
    command.add_argument("--no-closure", action="store_true")
    command.add_argument("--timeout-s", type=float, default=600.0)
    command.add_argument("--max-original-cells", type=int, default=10_000_000)
    command.add_argument("--max-chain-terms", type=int, default=500_000)
    command.add_argument("--max-work", type=int, default=5_000_000)
    command.add_argument("--max-basis-columns", type=int, default=200_000)
    command.add_argument("--max-basis-entries", type=int, default=5_000_000)
    command.add_argument("--max-new-cells", type=int, default=2_000)
    command.add_argument("--max-relative-rows", type=int, default=10_000)
    command.add_argument("--max-relative-work", type=int, default=2_000_000)
    command.add_argument("--max-examined-region-cells", type=int, default=250_000)
    command.add_argument("--reference-max-grid-cells", type=int, default=30_000)
    command.add_argument("--reference-max-old-cells", type=int, default=20_000)
    command.add_argument("--reference-max-boundary-rows", type=int, default=30_000)
    command.add_argument("--max-reference-queries", type=int, default=3)
    command.add_argument("--compare-component-reference", action="store_true")
    args = parser.parse_args()
    if args.command == "_worker":
        return _worker(args.config, args.result)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
