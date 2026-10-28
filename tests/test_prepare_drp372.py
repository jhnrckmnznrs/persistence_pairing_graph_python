"""DRP-372 frozen-cohort adapter: HDF5 MAT fixture and provenance checks."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "prepare_drp372_cohort.py"
spec = importlib.util.spec_from_file_location("prepare_drp372_cohort", SCRIPT)
drp = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(drp)


def build_fixture(tmp_path):
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "drp372_256"
    folder = root / "binary"
    folder.mkdir(parents=True)
    sample = "339_02"
    # MATLAB-v7.3 HDF5 variable bin, 256^3, binary solid by default.
    # Center pore shell: produces a true H2 void boundary in 12^3 crop.
    mat = folder / f"{sample}_256.mat"
    with h5py.File(mat, "w") as h:
        v = h.create_dataset(
            "bin", shape=(256, 256, 256), dtype="uint8",
            chunks=(32, 32, 32), fillvalue=1)
        v[122:134, 122:134, 122:134] = 0
        v[124:132, 124:132, 124:132] = 1
    csvpath = tmp_path / "complete_pairs.csv"
    with csvpath.open("w", newline="", encoding="utf-8") as f:
        fields = ["sample", "inclusion_status", "binary_present",
                  "lbm_present", "project_id",
                  "conservative_validation_group", "binary_sha256",
                  "permeability_values_json"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({
            "sample": sample, "inclusion_status": "included_complete_pair",
            "binary_present": "True", "lbm_present": "True",
            "project_id": "339", "conservative_validation_group": "339",
            "binary_sha256": drp.sha256(mat),
            "permeability_values_json": json.dumps(
                {"permeability (lus)": "1.25e-12"}),
        })
    return root, csvpath, mat


def test_native_crop_preserves_orientation_provenance_and_phase_values(tmp_path):
    root, csvpath, mat = build_fixture(tmp_path)
    out = tmp_path / "prepared"
    result = drp.generate(
        cohort_csv=csvpath,root=root,output=out,
        limit=1,crop_sizes=(12,),origins=("center","quarter"),
        filtrations=("binary",))
    assert result["samples"] == 1
    assert result["crops"] == 2
    assert result["permeability_with_valid_source_unit"] == 2
    manifest = json.loads((out / "cohort_manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert len(manifest["volumes"]) == 2
    assert manifest["volumes"][0]["group"] == "339"
    assert manifest["volumes"][0]["voxel_spacing_mm"] is None
    crop = np.load(out / manifest["volumes"][0]["path"])
    assert crop.dtype == np.uint8 and crop.shape == (12, 12, 12)
    assert set(np.unique(crop)).issubset({0,1})
    assert crop[0,0,0] == 0 and crop[5,5,5] == 1
    assert drp.sha256(out/manifest["volumes"][0]["path"]) == (
        manifest["volumes"][0]["sha256"])
    with (out / "crop_provenance.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["original_mat_sha256"] == drp.sha256(mat)
    assert rows[0]["origin_zyx"] == "[122, 122, 122]"
    assert float(rows[0]["permeability_lus"]) == 1.25e-12
    assert rows[0]["filtration"] == "binary_lower_star"
    with pytest.raises(FileExistsError):
        drp.generate(cohort_csv=csvpath,root=root,output=out,
                     limit=1,crop_sizes=(12,),origins=("center",),
                     filtrations=("binary",))


def test_adapter_pilot_runs_complete_ppg_audit_on_h2_shell(tmp_path):
    root, csvpath, _ = build_fixture(tmp_path)
    prepared = tmp_path / "prepared"
    drp.generate(cohort_csv=csvpath,root=root,output=prepared,
                 crop_sizes=(12,),origins=("center",),limit=1,
                 filtrations=("sedt",), levels_per_voxel=32,
                 sedt_source="recompute")
    script=Path(__file__).resolve().parents[1]/"benchmarks"/"certificate_audit.py"
    run=subprocess.run([
        sys.executable,str(script),"run",
        "--manifest",str(prepared/"cohort_manifest.json"),
        "--output",str(tmp_path/"audit"),
        "--radii","0","1","2","4",
        "--max-events-per-dimension","1",
        "--no-global",
        "--max-reference-queries","1",
        "--reference-max-grid-cells","30000",
        "--timeout-s","90",
    ],capture_output=True,text=True,timeout=100,check=False)
    assert run.returncode==0,run.stdout+"\n"+run.stderr
    rows=[json.loads(x) for x in (tmp_path/"audit"/"339_02__s12__center__sedt.jsonl")
          .read_text().splitlines()]
    events=[x for x in rows if x["kind"]=="event"]
    assert any(x["dimension"]==2 and x["status"]=="ok" for x in events)
    regions=[x for x in rows if x["kind"]=="region"]
    assert any(x["feasible"] for x in regions if x["status"]=="ok")
    assert any(x.get("verified") is True for x in regions)
    assert all(x["status"]=="ok" for x in regions)
    summary=json.loads((tmp_path/"audit"/"summary.json").read_text())
    assert summary["status"]=="complete"
    assert summary["verified_reconstructions"]>=1


def test_cohort_filters_and_hash_checks(tmp_path):
    root, csvpath, mat = build_fixture(tmp_path)
    with pytest.raises(ValueError,match="absent"):
        drp.generate(cohort_csv=csvpath,root=root,
                     output=tmp_path/"wrong",sample_ids=["777_99"],
                     filtrations=("binary",))
    with pytest.raises(ValueError,match="checksum"):
        with csvpath.open("r",newline="") as f:
            rows=list(csv.DictReader(f))
        with csvpath.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys())
            w.writeheader()
            rows[0]["binary_sha256"]="0"*64
            w.writerows(rows)
        drp.generate(cohort_csv=csvpath,root=root,
                     output=tmp_path/"wrong_hash",filtrations=("binary",))


def test_signed_edt_phase_polarity_and_historical_encoding():
    scipy = pytest.importorskip("scipy")
    full = np.zeros((5,5,5),dtype=np.uint8)
    full[2,2,2]=1
    q = drp.signed_edt_uint16(full,levels_per_voxel=32)
    assert q.dtype == np.uint16
    assert int(q[2,2,2]) == 32768 + 32
    assert int(q[2,2,1]) == 32768 - 32
    assert int(q[2,2,0]) == 32768 - 64
    assert int(q.min()) < 32768 < int(q.max())
    with pytest.raises(ValueError,match="both pore and solid"):
        drp.signed_edt_uint16(np.zeros((4,4,4),dtype=np.uint8),levels_per_voxel=32)


def test_sedt_requires_explicit_verified_scale(tmp_path):
    root, csvpath, _ = build_fixture(tmp_path)
    with pytest.raises(ValueError, match="levels-per-voxel"):
        drp.generate(
            cohort_csv=csvpath, root=root, output=tmp_path/"bad_sedt",
            limit=1, crop_sizes=(12,), origins=("center",),
            filtrations=("sedt",))
    input_mask = np.zeros((5,5,5), dtype=np.uint8)
    input_mask[2,2,2] = 1
    q = drp.signed_edt_uint16(input_mask, levels_per_voxel=64)
    assert int(q[2,2,2]) == 32768+64
    assert int(q[2,2,1]) == 32768-64


def make_saved_stage_e_fixture(root, mat, *, wrong_hash=False):
    """Use a synthetic 256^3 TIFF with known correct pore-negative signs."""
    tifffile = pytest.importorskip("tifffile")
    h5py = pytest.importorskip("h5py")
    with h5py.File(mat,"r") as h:
        binary=np.asarray(h["bin"])
    # This fixture checks reuse/provenance and orientation, not the
    # numerical Euclidean distance computation tested separately.
    enc=np.where(binary==0,32704,32832).astype(np.uint16)
    folder=root/"sedt_stage_e"/"sedt"
    folder.mkdir(parents=True,exist_ok=True)
    tif=folder/"339_02_sedt_u16.tif"
    tifffile.imwrite(tif,enc,photometric="minisblack")
    with (root/"sedt_stage_e"/"sedt_manifest.csv").open(
            "w",newline="",encoding="utf-8") as stream:
        fields=["sample","status","input_binary_sha256_observed",
                "levels_per_voxel","zero_code","output_path",
                "output_sha256","encoded_min","encoded_max",
                "family_validation_group","project_validation_group",
                "primary_outer_fold"]
        w=csv.DictWriter(stream,fieldnames=fields)
        w.writeheader()
        w.writerow({
            "sample":"339_02","status":"ok",
            "input_binary_sha256_observed":drp.sha256(mat),
            "levels_per_voxel":"64.0","zero_code":"32768",
            "output_path":"/home/john-rick/Desktop/drp372/drp372_256/sedt_stage_e/sedt/339_02_sedt_u16.tif",
            "output_sha256":("0"*64 if wrong_hash else drp.sha256(tif)),
            "encoded_min":int(enc.min()),"encoded_max":int(enc.max()),
            "family_validation_group":"family_339",
            "project_validation_group":"DRP339","primary_outer_fold":"2",
        })
    return tif


def test_saved_stage_e_reuses_old_tiff_and_rebases_stale_paths(tmp_path,monkeypatch):
    root, csvpath, mat = build_fixture(tmp_path)
    saved=make_saved_stage_e_fixture(root,mat)
    def forbid_recompute(*args,**kwargs):
        raise AssertionError("Unexpected full-volume EDT recomputation")
    monkeypatch.setattr(drp,"signed_edt_uint16",forbid_recompute)
    out=tmp_path/"using_saved_stage_e"
    report=drp.generate(
        cohort_csv=csvpath,root=root,output=out,
        limit=1,crop_sizes=(12,),origins=("center",),
        filtrations=("sedt","binary"),levels_per_voxel=64)
    assert report["samples"]==1 and report["crops"]==2
    assert report["sedt_source"]=="stage-e"
    manifest=json.loads((out/"cohort_manifest.json").read_text())
    assert len(manifest["volumes"])==2
    assert all(e["group"]=="family_339" for e in manifest["volumes"])
    sediment=next(e for e in manifest["volumes"] if e["id"].endswith("__sedt"))
    crop=np.load(out/sediment["path"])
    assert crop.dtype==np.uint16
    assert crop[0,0,0]==32704 and crop[5,5,5]==32832
    with (out/"crop_provenance.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    sedt_row=next(e for e in rows if e["id"].endswith("__sedt"))
    assert sedt_row["stage_e_original_tiff_path"]==str(saved)
    assert sedt_row["stage_e_original_tiff_sha256"]==drp.sha256(saved)
    assert sedt_row["family_validation_group"]=="family_339"
    assert sedt_row["historical_primary_outer_fold"]=="2"
    assert not any("Desktop" in e["cropped_npy_path"] for e in rows)


def test_saved_stage_e_rejects_changed_hash_or_wrong_scale(tmp_path):
    root, csvpath, mat = build_fixture(tmp_path)
    make_saved_stage_e_fixture(root,mat,wrong_hash=True)
    with pytest.raises(ValueError,match="SHA-256 mismatch"):
        drp.generate(
            cohort_csv=csvpath,root=root,output=tmp_path/"bad_hash",
            limit=1,crop_sizes=(12,),origins=("center",),
            filtrations=("sedt",),levels_per_voxel=64)
    with pytest.raises(ValueError,match="levels_per_voxel"):
        drp.generate(
            cohort_csv=csvpath,root=root,output=tmp_path/"wrong_scale",
            limit=1,crop_sizes=(12,),origins=("center",),
            filtrations=("sedt",),levels_per_voxel=32)


def test_saved_stage_e_rejects_wrong_axis_alignment(tmp_path):
    root, csvpath, mat = build_fixture(tmp_path)
    tif=make_saved_stage_e_fixture(root,mat)
    tifffile=pytest.importorskip("tifffile")
    original=tifffile.imread(tif)
    altered=original.copy()
    # Shift the phase-matched shell by one voxel while keeping TIFF valid.
    altered=np.roll(altered,1,axis=0)
    tifffile.imwrite(tif,altered,photometric="minisblack")
    manifest=root/"sedt_stage_e"/"sedt_manifest.csv"
    rows=list(csv.DictReader(manifest.open(newline="")))
    rows[0]["output_sha256"]=drp.sha256(tif)
    with manifest.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError,match="phase/axis mismatch"):
        drp.generate(cohort_csv=csvpath,root=root,output=tmp_path/"axis_bad",
                     limit=1,crop_sizes=(12,),origins=("center",),
                     filtrations=("sedt",),levels_per_voxel=64)
