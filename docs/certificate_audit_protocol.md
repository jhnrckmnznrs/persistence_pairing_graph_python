# PPG certificate audit: reproducible real-volume protocol

This document describes the benchmark workflow for **Manuscript II**. It
makes no empirical claims about any cohort until those actual files have
been processed. Synthetic smoke tests verify the machinery only.

## 1. Input manifest

Use `benchmarks/certificate_audit.py` with a JSON manifest:

```json
{
  "schema_version": 1,
  "volumes": [
    {
      "id": "porous_0001",
      "path": "data/porous_0001.tif",
      "group": "source_A",
      "split": "benchmark",
      "voxel_spacing_mm": [0.025, 0.025, 0.025],
      "sha256": null
    },
    {
      "id": "bone_example",
      "path": "data/bone_example.npy",
      "group": "microct",
      "voxel_spacing_mm": [0.02, 0.02, 0.02],
      "event_ids": [0]
    }
  ]
}
```

- Each `path` resolves relative to the manifest's directory.
- Accepted data are **single** 2D/3D uint8/uint16 TIFF or NPY arrays. TIFF
  stacks should be verified before processing. Multi-channel and floating
  arrays are rejected, not silently collapsed or quantized.
- `id` must be unique and filesystem-safe; `group` and `split` are
  optional provenance labels. File contents are SHA-256 hashed and a supplied
  expected `sha256` is checked before computation.
- `voxel_spacing_mm` is descriptive metadata, not used to resample or
  alter the filtration. Coordinate bounds are in the package's **doubled
  cubical grid**, so a radius of `r` input voxels expands each axis by
  `2*r` grid coordinates. For anisotropic grids this does NOT represent
  a fixed physical-radius ball.
- The original raw images and full certificate cell IDs are **not copied**
  to the audit output, but the output may still contain locally identifying
  filenames, group labels, and image checksums. Do not publish it without
  checking the data-use agreement.

## 2. Execute

```sh
python -m pip install -e '.[images,test]'
python benchmarks/certificate_audit.py run \
  --manifest /path/to/cohort_manifest.json \
  --output /path/to/certificate_audit \
  --radii 0 1 2 4 8 16 \
  --max-events-per-dimension 3 \
  --decision-repeats 3 \
  --timeout-s 600 \
  --reference-max-grid-cells 30000 \
  --max-reference-queries 3 \
  --compare-component-reference
```

Each manifest entry runs in a fresh subprocess, so peak RSS and failures
are separated by volume. An external timeout terminates a stuck process.
If the output directory is nonempty, the workflow refuses to overwrite
unless `--overwrite` is specified.

**Important:** the worker is resource-bounded by candidate-cell,
work, and overall original-grid safety caps, but it currently has no
hard memory kill threshold. Use the operating system's memory limit,
job scheduler, or container/cgroup limits on shared servers.

## 3. Event and region selection

By default the tool selects, independently for H1 and H2, the
`max-events-per-dimension` finite positive-length intervals with largest
persistence, breaking ties by event filtration value, compressed birth index,
then node ID. The deterministic selection is applied separately to each
volume. Use explicit `event_ids` only when the IDs are already validated
for that exact input and software revision.

Each region is an axis-aligned box centered at the death **critical cell**,
not the centroid of an existing filling or an optimized region. Boxes are
nested for increasing voxel radii. An optional global box is included.
Only **new (q+1)-cells entering at the exact event value** are constrained
by the box; the allowed old correction is global within K_{<t}.

No region is chosen based on the result of a previously reconstructed
filling. This avoids the artificial positive-selection bias of taking a
known witness's own support as the candidate.

For nested regions T1 contained in T2:
```
feasible(T1) => feasible(T2).
```
The audit checks this for completed queries. A box that exceeds the
candidate or region work cap is recorded as an ERROR, not assumed
infeasible. `global` may consequently be unavailable at large sizes.

## 4. Measurements and independent checks

The workflow records:

1. File checksum, shape, dtype, spacing metadata, group/split.
2. PPG preparation and oracle-index time; compressed generator count,
   cancellation-pair count, process peak RSS.
3. Stable node ID, H1/H2 dimension, birth/death and persistence, relation
   support size, initial replay witness time and witness support size.
4. Optional observed reverse-replay dependency closure (number reached,
   replayed, arcs, depth, and fraction); this excludes preprocessing.
5. Each box's exact bounds and candidate enumeration time, number of
   new cells, repeated **decision-only** times, feasibility, rank/relative
   kernel statistics, lazy cocycle obstruction size.
6. For the first feasible box, optional complete original-grid filling
   reconstruction and independent checks of d(a+z)=c, all old cells in
   K_{<t}, all new filling cells within the permitted set, and a SHA-256
   digest of the filling indices.
7. For small cases only, an independent full old-cell reference computation
   and exact feasibility agreement. Reference cases are limited by both
   maximum original grid size and per-volume reference query count.
8. Optional incidence-component reference comparisons (with
   `--compare-component-reference`) using the same permitted new cells and
   bounded reference budget; this is distinct from the full old-cell scan.
9. The first sufficient *tested* nested radius per event, the number of
   completed region queries, and whether all requested regions were decided.
   This is not a globally smallest geometric region.
10. Explicit statuses for unsupported, skipped-budget, timed-out, or
   failed queries and workers.

**Do not report a reference comparison on a large case as passed if it
was skipped.** Do not interpret a fast decision-only query as the cost of
producing a complete certificate, or per-query times as end-to-end speed.

The current benchmark executes sparse checks before full-reference checks
and shares caches within a volume. To claim statistically stable runtime
speedups, a larger controlled follow-up should randomize method order or
run each method in independent processes and repeat volumes/queries.

## 5. Output and failure preservation

Each run writes:

- `configuration.json`: resolved manifest and options, Python/NumPy/platform,
  installed package version and Git revision when available.
- `<volume_id>_worker_config.json`: exact per-volume worker inputs.
- `<volume_id>.jsonl`: append-only stage, event, region and failure records.
- `events.csv` and `regions.csv`: flat summaries; nested fields encoded
  as JSON strings.
- `summary.json`: volumes, selected events, region decisions,
  reconstructions, independent reference comparisons and failures.

A worker crash, timeout, wrong checksum, invalid requested event, or
failed exact validation produces nonzero exit status at the parent level;
already flushed per-volume records are retained. A completely constant
volume can legitimately have **zero eligible finite H1/H2 death events**,
which must remain visible rather than be replaced by a manufactured event.

## 6. Evaluation matrix before submitting Manuscript II

**Pilot (32³–64³).** Check sample parsing, event selection, q=1/q=2
coverage, exactness and reference agreement on a tractable subset.
Use several distinct regions per event and include empty/insufficient
boxes, sufficient boxes, and the global region when possible.

**Scale-out (96³–128³).** Use the same event policy and limits, track
query and preparation time, process RSS, dependency closure, witness
size, and skipped/failed rate. Avoid calling skipped reference checks
successful.

**Real porous-media cohort.** Fix source/group stratification, record
any outcome and geometry feature separately from the topology audit;
evaluate physical spatial interpretation and predictive utility later,
using grouped validation. The audit itself does not fit a predictive
model or compute a claim of predictive improvement.

**Bone micro-CT external cohort.** Explicitly document quantization,
segmentation, cropping, and voxel spacing. Use the cohort only after
the principal PPG real-data claims are established, with inference
appropriately cautious for a small number of specimens.

**Manuscript criteria.** Zero unexplained disagreements in all
independently checked cases, published counts of failures and skipped
queries, statistical uncertainty for computational comparisons,
preprocessing-inclusive costs, and clear distinction between
mathematical correctness and geometric/scientific utility.

## 7. Current limitations

- Benchmarks run per volume in one process, with repeated queries sharing
  caches and a canonical reduction. The selection covers selected long
  persistence intervals, not a representative random sample of all events.
- Nested boxes centered at death critical cells are a *specified region
  family*. The first sufficient box in the family is not a globally
  minimum-area or minimum-support filling.
- The certificate solver uses field F2 and the package's lower-star
  (minimum-incident-top-cell) sublevel filtration, not a signed-distance
  convention unless the supplied integer array already encodes it.
- Input hashes document bytes; independently resampled data may have
  different hashes even if the underlying specimen is the same.
- `ru_maxrss` provides process peak RSS in KiB on Linux, not allocated
  array bytes or perfect stage attribution. Scratch-disk usage and physical
  CPU utilization are not measured by the initial workflow.
- There is no real-dataset experiment bundled here; TIFF fixtures are
  deliberately synthetic correctness examples.
