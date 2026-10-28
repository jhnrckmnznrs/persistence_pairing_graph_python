# DRP-372 digital-rock PPG rerun — frozen 119-pair cohort

This pilot returns to the **same DRP-372 source** used for the earlier
August 2026 project (119 complete 256³ geometry/LBM.csv pairs), not a
replacement digital rock dataset. It uses the existing verified 0=pore,
1=solid phase semantics and the historical DRP-372 signed-distance
filtration (SEDT), using the same offset/scale uint16 encoding as the
earlier Rust PPG experiment. No permeability result is fabricated.

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

## Required preflight: verify historical Stage E before preparing any new SEDT

The already verified Stage G freeze has 119 complete specimens with
matching Stage F node and edge hashes, **but it records
`levels_per_voxel=64.0` and `zero_code=32768`**. Earlier prototype
adapter revisions mistakenly assumed SEDT scale 32. The source Stage E
script and actual encoded files must settle this discrepancy.

Run the **read-only** preflight from this repository:

```bash
python benchmarks/inspect_drp372_reuse.py \\
  --workspace "$HOME/temp/drp372" \\
  --output "$HOME/temp/drp372/ppg_reuse_preflight.json"
```

It summarizes saved Stage E/F/G manifest schemas, sample-set consistency,
original SEDT source-code constants and project/family fold crossings.
It does **not** open any full 256³ array or modify old results. Paths
embedded in older Stage G summaries may refer to a former
`~/Desktop/drp372` location; the preflight resolves files in the
current workspace instead.

**Scale and saved files verified by your preflight.** The frozen 119-case
Stage E manifest and original source script both confirm
`levels_per_voxel=64.0`, `zero_code=32768`, and quantization step
`1/64` voxel. The new adapter's **default** is
`--sedt-source stage-e`, which reuses the existing
`sedt_stage_e/sedt/<sample>_sedt_u16.tif` after verifying the original
source MAT SHA-256, saved TIFF SHA-256, encoded extrema, dtype, shape
and crop-level sign/axis agreement with the binary source.
The obsolete absolute `~/Desktop/drp372` path is never used to
retrieve the TIFF. A mismatch fails rather than silently recomputing.

Only `--sedt-source recompute` recomputes the full 256³ SEDT and it
must be explicitly requested; recomputation is **not** the recommended
reproduction path. The new certificate backend still builds its own
PPG/reduction **for cropped inputs**. Stage F outputs are retained
for original full-rock comparisons, not assumed to contain a reusable
Python filtered contraction.

**Validation-fold caution:** Stage G's project-group labels DRP16,
DRP317, and DRP374 occur in multiple primary folds. This is not
project-disjoint validation as recorded; check the original grouping
script for whether folds were built around a different family/grouping
criterion before reusing earlier model scores.

**Your verified local paths (from the recovered workspace):**

```text
~/temp/drp372/drp372_256/cohort_qc_v1_1/complete_pairs.csv
~/temp/drp372/drp372_256/sedt_stage_e/sedt_manifest.csv
~/temp/drp372/drp372_256/sedt_stage_e/sedt/<sample>_sedt_u16.tif
```

The root `~/temp/drp372/drp372_256` contains the frozen 119-case
cohort. Stage E, Stage F and Stage G each report 119 records, and the
saved Stage F node/edge hashes match across the full cohort.

## Stage 1: lossless original-voxel pilot crops

```sh
python -m pip install -e '.[images,rock,test]'

python benchmarks/prepare_drp372_cohort.py \
  --root "$HOME/temp/drp372/drp372_256" \
  --cohort-csv "$HOME/temp/drp372/drp372_256/cohort_qc_v1_1/complete_pairs.csv" \
  --output "$HOME/temp/drp372/ppg_stage_e_crops_pilot" \
  --limit 3 \
  --sizes 12 24 \
  --origins center quarter \
  --filtrations sedt binary \
  --sedt-source stage-e \
  --levels-per-voxel 64
```

This creates:

- `cohort_manifest.json`: directly accepted by
  `benchmarks/certificate_audit.py`.
- `volumes/<id>.npy`: native 3D SEDT uint16 crops and/or binary uint8 controls.
- `crop_provenance.csv`: sample/source file hashes, input coordinate
  offsets, phase labels, pore fraction, source-group labels and the
  strictly named `permeability (lus)` value where available.
- `preparation_summary.json`: selected samples, crop sizes, and a
  warning that crops **do not preserve full-rock topology**.

No interpolation, majority downsampling, CT intensity normalization,
phase inversion or resegmentation is performed.

The output `crop_provenance.csv` also stores each Stage E source TIFF
hash, current path, its historical family/project validation groups,
and the original primary outer fold. For future modeling, keep the
**family group** as the original nonoverlapping fold unit; the
`DRP16`, `DRP317` and `DRP374` project labels cross folds.
The audit itself makes no predictive or project-held-out inference.

**Historical primary filtration:** These are the original Stage E
encoding equations. In the recommended `stage-e` mode, the transform
below is **not recomputed**; it is already stored in the verified
full-volume TIFF:

```python
s = scipy.ndimage.distance_transform_edt(bin == 1) \
    - scipy.ndimage.distance_transform_edt(bin == 0)
q = np.uint16(np.clip(np.rint(32768 + 64*s), 0, 65535))
```

Negative `s` is pore and positive `s` is solid. The primary
`sedt` outputs are uint16, and the `binary` outputs are separate
uint8 controls. The offset 32768 and explicitly verified scale `S` should match the
historical Stage E encoding. The original script must also confirm the
sign convention, array axes and distance-transform boundary semantics. No per-sample min-max normalization is applied. The metadata can be inspected
without regenerating features, and the cropped TIFF values are bitwise
copied to the pilot NPY arrays.
Because the **full-volume** transform is computed first, the interior
crop distances remain those of the complete rock. However, the
resulting cropped cubical complex still has artificial topological
boundaries, which should be acknowledged in analysis.

A full 256³ Euclidean-distance computation can temporarily allocate
several hundred MiB. Prepare samples sequentially and keep the
preparation memory budget separate from the PPG benchmark's peak RSS.

The sample selection is deterministic: sorted complete-pair sample IDs,
restricted by `--sample` when given, then `--limit` (default 5).
This is a technical pilot, not a representative selection for scientific
inference. Source project IDs serve as conservative grouping labels.

One can select specific samples from the already validated cohort:

```sh
python benchmarks/prepare_drp372_cohort.py \
  --root /data/drp372_256 \
  --cohort-csv "$HOME/temp/drp372/drp372_256/cohort_qc_v1_1/complete_pairs.csv" \
  --output "$HOME/temp/drp372/ppg_stage_e_crops_pilot" \
  --sample 10_01 \
  --limit 1 \
  --sizes 12 24 \
  --sedt-source stage-e \
  --levels-per-voxel 64
```

The preflight explicitly confirms that `10_01` belongs to the frozen
119-case cohort. Additional IDs should be selected from the audited CSV.

## Stage 2: exact PPG certificate audit

Start small and use bounded nested regions:

```sh
python benchmarks/certificate_audit.py run \
  --manifest "$HOME/temp/drp372/ppg_stage_e_crops_pilot/cohort_manifest.json" \
  --output "$HOME/temp/drp372/ppg_stage_e_crops_audit" \
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

## If the first real-rock crop contains no positive-persistence events

On 28 October 2026 the **actual Stage E SEDT crop** of
`10_01`, size 12³, centered at the volume midpoint, produced:

```json
{"volumes":1,"events":0,"regions":0,
 "verified_reconstructions":0,"status":"complete"}
```

This is **not** a failed relative-filling query and **does not**
validate certificate reconstruction on the real rock. The selector
requires finite positive-length H1/H2 bars; small boxes can lack
such bars even when the full rock has complicated topology.

The audit now reports `event_inventories` (finite positive, finite
zero, and essential counts by dimension), filtration min/max, distinct
uint16 levels and numbers of codes below/at/above SEDT zero code
32768. Examine `summary.json` and each per-volume JSONL record
to distinguish a homogeneous/uninteresting crop from the selector's
positive-persistence restriction.

Do not alter the SEDT quantization scale or interpret zero events
as evidence that the full original 256³ rock has trivial homology.

A small pre-registered **scale and location diagnostic** for specimen
`10_01` is:

```bash
python benchmarks/prepare_drp372_cohort.py \\
  --root "$HOME/temp/drp372/drp372_256" \\
  --cohort-csv "$HOME/temp/drp372/drp372_256/cohort_qc_v1_1/complete_pairs.csv" \\
  --output "$HOME/temp/drp372/ppg_stage_e_crops_multiscale" \\
  --sample 10_01 --limit 1 \\
  --sizes 24 32 --origins center quarter three_quarter \\
  --filtrations sedt --sedt-source stage-e \\
  --levels-per-voxel 64

python benchmarks/certificate_audit.py run \\
  --manifest "$HOME/temp/drp372/ppg_stage_e_crops_multiscale/cohort_manifest.json" \\
  --output "$HOME/temp/drp372/ppg_stage_e_audit_multiscale" \\
  --radii 0 1 2 4 --no-global \\
  --max-events-per-dimension 2 --decision-repeats 2 \\
  --max-original-cells 1000000 --max-new-cells 2000 \\
  --max-reference-queries 1 --timeout-s 600
```

These are six *non-independent* subvolume cases from one specimen:
24³ and 32³ at three predetermined locations. Each cropped
filtration uses the bitwise-saved full-256³ SEDT. This is a software
and local-topology diagnostic, not evidence of predictive permeability
associations. If several are still zero-event, expand to **different
source family groups** using the frozen 119-sample cohort while keeping
the specimen selection independent of target permeability.

Do not delete or overwrite the original successful 12³ audit output.

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
4. Keep the historical **SEDT uint16** filtration as the primary
   reproduction and the raw binary (0 pore / 1 solid) filtration as a
   separate control. Never combine their PPG node/edge counts into
   one unlabelled feature dataset. Quantization, SciPy distance boundary
   convention, and crop origin are mandatory metadata.
5. Permeability is an independent whole-rock LBM target, and a small
   crop does not have a physically measured permeability. Therefore,
   do not report crop-level correlations as independent permeability
   validation. Aggregate descriptors to the specimen and use group-aware
   cross-validation.
6. Do not submit empirical improvements to Manuscript II until actual
   DRP-372 experiment outputs, failures, peak RSS, calibration and
   baseline comparisons are available.

## Previous full-volume DRP-372 PPG results to preserve

The original 119-pair full-volume experiment (Rust
`cubical_persistence --algorithm stream-block`) used the encoded SEDT,
not the raw 0/1 image as the sole primary filtration. A recorded example
had 135,005,697 cubical cells, 36,873 compressed generators, 54.11 s
runtime and 689.22 MiB peak RSS. These figures were measured in the
earlier Rust environment; they must not be attributed to the current
Python oracle or its 12³/24³ pilot crops.

The previous whole-cohort feature material included:
- `drp372_256/features_stage_g/combined_features.csv`
- `drp372_256/features_stage_g/core_feature_sets_v2.json`
- `drp372_256/fpg_stage_f/` PPG output files.

The earlier feature aggregation had 648 descriptors (4 geometry,
224 persistence, 116 pure graph, 304 coupled), with a compact 21-feature
persistence descriptor block across H0, H1 and H2. Prior
project/family-grouped log10 lattice-permeability models included
Ridge RMSE 0.7665 / R² 0.4529 and PLSR RMSE 0.6585 / R² 0.5962.
These historical values must be reverified against the frozen saved
feature tables and split definitions before being republished.

A crucial negative control from September 2026: shuffled
persistence-attribute coupling and event-constrained PPG target rewiring
were not predictive disadvantages relative to the observed PPG.
For example, an observed coupled model RMSE around 0.8653 was not
better than its persistence-attribute surrogate median around 0.8375.
The extra localization/certificate features should be compared not
only against persistence diagrams and graph summaries but also against
these **structurally constrained null models**. Otherwise, finding
non-random PPG structure would not establish predictive usefulness.

The new benchmark workflow measures exact spatial certificates and
runtime; it does not yet refit any model or rerun these historical
nulls. After recovering the existing whole-cohort features, we should
join them by **original sample ID** and retain project-grouped
splits. Subcrops inherit the whole-rock permeability target only as
specimen metadata: they do not constitute new independent permeability
measurements.

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
