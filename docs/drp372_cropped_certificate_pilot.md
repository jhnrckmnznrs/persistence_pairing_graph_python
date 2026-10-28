# DRP-372 digital-rock PPG rerun — frozen 119-pair cohort

This pilot returns to the **same DRP-372 source** used for the earlier
August 2026 project (119 complete 256³ geometry/LBM.csv pairs), not a
replacement digital rock dataset. It uses the existing verified 0=pore,
1=solid phase semantics and the PPG package's cubical sublevel
filtration. No permeability result is fabricated.

## Prerequisites: the frozen local cohort

The original August 2026 cohort scripts produced:

```text
drp372_256/
  binary/
    339_02_256.mat
    ...
  targets/
    339_02_256_LBM.csv
    ...
  cohort_qc/
    catalogue_manifest.csv
    complete_pairs.csv
    excluded_catalogue.csv
    ...
```

The frozen cohort originally had 138 expected 256³ catalogue records,
133 locally acquired binary files, and 119 complete geometry–standard
LBM.csv pairs. **Do not replace the 119 sample list with all accessible
geometries**, because the extra cases do not have matching validated
permeability targets. Use the existing
`drp372_finalize_cohort_v1_1.py` outputs and the correct source archive.

The source DRP-372 project is the Digital Rocks Portal / Digital Porous
Media Portal collection associated with the Large-simulation-dataset
paper. Download acquisition and QC were previously handled by
`drp372_selective_downloader.py` and
`drp372_finalize_cohort_v1_1.py`. They are separate user research
scripts; **the PPG repo does not bundle the 119 binary MAT files**.

## Stage 1: lossless original-voxel pilot crops

```sh
python -m pip install -e '.[images,rock,test]'

python benchmarks/prepare_drp372_cohort.py \
  --root /data/drp372_256 \
  --cohort-csv /data/drp372_256/cohort_qc/complete_pairs.csv \
  --output /data/ppg_drp372_pilot \
  --limit 5 \
  --sizes 12 24 \
  --origins center quarter
```

This creates:

- `cohort_manifest.json`: directly accepted by
  `benchmarks/certificate_audit.py`.
- `volumes/<id>.npy`: 2D/3D uint8 arrays (here native 3D crops).
- `crop_provenance.csv`: sample/source file hashes, input coordinate
  offsets, phase labels, pore fraction, source-group labels and the
  strictly named `permeability (lus)` value where available.
- `preparation_summary.json`: selected samples, crop sizes, and a
  warning that crops **do not preserve full-rock topology**.

No interpolation, majority downsampling, CT intensity normalization,
phase inversion or resegmentation is performed.

The sample selection is deterministic: sorted complete-pair sample IDs,
restricted by `--sample` when given, then `--limit` (default 5).
This is a technical pilot, not a representative selection for scientific
inference. Source project IDs serve as conservative grouping labels.

One can select specific samples from the already validated cohort:

```sh
python benchmarks/prepare_drp372_cohort.py \
  --root /data/drp372_256 \
  --cohort-csv /data/drp372_256/cohort_qc/complete_pairs.csv \
  --output /data/ppg_drp372_pilot \
  --sample 339_02 57_04 \
  --limit 2 \
  --sizes 12 24
```

Replace those example sample IDs with IDs present in *your* frozen
`complete_pairs.csv`. Do not assume any demonstration ID belongs to
the 119-pair complete-case cohort without checking the CSV.

## Stage 2: exact PPG certificate audit

Start small and use bounded nested regions:

```sh
python benchmarks/certificate_audit.py run \
  --manifest /data/ppg_drp372_pilot/cohort_manifest.json \
  --output /data/ppg_drp372_audit \
  --radii 0 1 2 4 \
  --no-global \
  --max-events-per-dimension 2 \
  --decision-repeats 3 \
  --max-original-cells 1000000 \
  --max-new-cells 2000 \
  --max-examined-region-cells 200000 \
  --reference-max-grid-cells 30000 \
  --max-reference-queries 2
```

The audit uses the complete PPG death relation for each selected event,
not a single outgoing edge. The relative solver allows arbitrary old
chain corrections in the full cropped sublevel complex, but restricts
new event-level filling cells to the prescribed region.

A large region can exceed the oracle's bound because a binary
filtration has many tied entering cells; this is recorded as a resource
error, not as an infeasible certificate. The pilot is not a proof that
the method scales to the full 256³ image.

Interpret the end-to-end results in
`<output>/summary.json`, `events.csv`, `regions.csv`, and the
per-volume JSONL audit files. Skipped independent full references are
not silently treated as verified.

## Stage 3: scientific protocol

After the software pipeline passes:

1. Audit all 119 original 256³ inputs in the frozen cohort for pairing,
   file hashes, coordinate order and source porosity.
2. Fix the evaluation sampling *before* looking at permeability. Include
   distinct source-project groups and avoid taking only easy-to-solve
   high-porosity samples.
3. Increase size in stages (12³, 24³, 32³, then 64³/96³ as memory allows).
   Track timings for PPG preparation, re-used coordinate decisions,
   complete chain reconstruction and verification.
4. Keep the original **binary** pore filtration as the primary
   reproduction. A derived distance-to-interface filtration may be
   informative but is a *separate sensitivity analysis* and must specify
   sign, quantization, voxel spacing, and crop-boundary treatment.
5. Permeability is an independent whole-rock LBM target, and a small
   crop does not have a physically measured permeability. Therefore,
   do not report crop-level correlations as independent permeability
   validation. Aggregate descriptors to the specimen and use group-aware
   cross-validation.
6. Do not submit empirical improvements to Manuscript II until actual
   DRP-372 experiment outputs, failures, peak RSS, calibration and
   baseline comparisons are available.

## Source and citation

Dataset project: <https://digitalporousmedia.org/published-datasets/tapis/projects/drp.project.published/drp.project.published.DRP-372>

Original dataset code/description:
<https://github.com/je-santos/Large-simulation-dataset>

Original project DOI:
<https://doi.org/10.17612/93pd-y471>

The collected sample geometry files are published scientific data; check
their source licence and the dataset citation requirements before releasing
derived crops or distributing binary files.

## Current status

This branch contains the adapter and synthetic HDF5 end-to-end tests.
**It does not include or claim to have processed the user's 119 original
MATLAB volumes.** Those source files exist in the user's previous
local data environment, but were not attached to this ChatGPT session.
