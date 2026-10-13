# Persistence Pairing Graph — Python

`persistence-pairing-graph` is a Python package for computing **persistence-pairing graphs (PPGs)** from grayscale 2D images and 3D image volumes. It also provides independent Betti-curve calculations, original-grid cycle representatives from both reference and streaming reductions, event-level graph summaries, per-edge geometric explanations, multiscale topology, and perturbation diagnostics.

The package computes cubical **sublevel** persistent homology over **F₂**. Each image pixel or voxel is a closed top-dimensional cube. Its lower-dimensional faces enter at the **minimum** intensity among incident pixels/voxels. A PPG vertex is a positive-length persistence interval (or essential class); a directed edge records a transition coefficient obtained from the specified canonical persistence reduction. **PPG edges depend on the selected reduction basis** and are not determined by the persistence barcode alone.

The required runtime dependency is NumPy. The Python implementation does not need Rust. Optional Numba acceleration is available for discrete filtration ordering, compressed-boundary emission, sparse reduction, and the independent Betti-curve solver.

## Install

```bash
python -m pip install -e '.[images,test]'
python -m pytest -q
```

For accelerated kernels:

```bash
python -m pip install -e '.[images,speed,test]'
```

Python 3.10+ is required. TIFF reading uses `tifffile`, and PNG/JPEG reading uses Pillow. These are optional extras because arrays can be passed directly to the Python API.

## Compute a persistence-pairing graph

```python
import numpy as np
from persistence_pairing_graph import compute_ppg

image = np.random.default_rng(7).integers(0, 32, (32, 32, 32), dtype=np.uint8)
graph = compute_ppg(image, algorithm="stream-block")
print(len(graph.nodes), graph.edge_count)
```

Process an image from the command line:

```bash
ppg volume.tif --algorithm stream-block
```

This produces `volume_persistence_pairing_graph_nodes.csv` and `volume_persistence_pairing_graph_edges.csv` with the same column conventions as the Rust reference implementation.

For larger volumes with sufficient RAM for the Numba runtime:

```bash
ppg volume.tif --algorithm stream-block \
    --accelerator numba --order-builder counting \
    --reduction-storage packed --node-storage compact \
    --stream-edges --event-summary --degrees
```

For memory-constrained workloads, a disk-backed order and contraction state can be combined with disk-backed reduction:

```bash
ppg volume.tif --algorithm stream-block \
    --order-format delta --metadata-storage disk \
    --reduction-storage disk --stream-edges --temp-dir /scratch
```

Scratch storage must be large enough and is deleted after normal completion or ordinary errors. Disk-backed contraction metadata reduces the need to keep the entire state array in *non-reclaimable* RAM; mapped pages can still contribute to reported process RSS, so it does not guarantee a lower measured RSS or fast random-access performance. **The complete pipeline is not fully out of core**, because compressed boundaries, indices, and graph event workspaces still need memory. A 256³ image exceeds the default 100-million-cubical-cell safety guard, even though the compact state representation can address that many cells when the guard is increased explicitly.

### Algorithm and storage choices

| Setting | Purpose |
| --- | --- |
| `algorithm="stream-block"` | Bucketed, discrete `uint8`/`uint16` equal-value contraction; compact boundaries |
| `algorithm="block"` | Materialized equal-value contraction, including finite float input |
| `algorithm="lazy"` | Uncompressed canonical reference reduction |
| `order_format="u32"` or `"delta"` | Direct 32-bit ordered IDs (faster) or delta-varint encoding (less scratch storage) |
| `order_builder="sort"` or `"counting"` | Stable plane sorts or compiled O(cells)-work counting scatter; counting requires Numba and `u32` order |
| `metadata_storage="memory"` or `"disk"` | Resident or memory-mapped contraction state |
| `reduction_storage="packed"`, `"disk"`, or `"python"` | Packed arrays, scratch-disk payloads, or reference Python lists |
| `node_storage="compact"` or `"python"` | Eight bytes per node endpoint pair or ordinary Python node records |
| `accelerator="python"` or `"numba"` | Pure Python or optional compiled numerical kernels |
| `log_mode="full"` or `"selective"` | Full canonical operation logging or selective two-pass operation logging |

`order_builder="auto"` selects counting scatter for Numba with `u32` ordering and the stable sorter otherwise. None of these options intentionally changes canonical persistence pivots or PPG edges. Experimental optimized kernels are regression-tested against reference data, but cross-language Rust parity still requires executing the bundled comparison harness with Rust installed.

## Experimental transition filling certificates

The research API `transition_filling_witness` reconstructs one **whole death-event relation** on the original doubled cubical grid. If a PPG node `u` dies at `t`, the boundary of the reported `(q+1)`-chain is exactly its lifted birth cycle plus the lifted birth cycles of **all** outgoing PPG targets (over F2). A single PPG edge does not normally possess its own filling certificate.

```python
from persistence_pairing_graph import transition_filling_witness

certificate = transition_filling_witness(image, source_node=2)
print(certificate.target_nodes, certificate.event_value)
print(certificate.filling_grid_indices)
print(certificate.filling_doubled_coordinates)
```

Two exact reference methods are available: `solver="pair-trace"` (default) finds a compressed filling from `R = DV`, maps its surviving-cell generators to original grid IDs, then corrects the boundary using **only recorded canceled death cells**; `solver="full-reference"` independently solves against all eligible original `(q+1)`-cell boundaries at the event. Both verify the original-grid boundary equation. The pair-trace path checks the strict-predeath obstruction in the compressed complex, justified by the filtered contraction theorem below, while the full-reference path checks it independently on the original grid. These fillings are not shortest, minimal-area, or generally unique.

The selective correction is a **sparse GF(2) basis**, indexed by nonzero original boundary IDs, not by a bit position for every cell in the original grid. For `pair-trace`, the full doubled-grid filtration and the full-grid bitset are never materialized solely to reconstruct the witness; cell birth values are checked only for selected filling cells. `full-reference` retains its deliberately small-grid bitset implementation. The default safety guard of **20,000 original cells** and **6,000 correction columns** still applies to both modes unless explicitly raised; `max_basis_entries` and `max_chain_terms` also bound memory. Neither mode is fully out of core, and high-complexity fillings can still be expensive. The canonical PPG remains basis-dependent, and fillings are additional nonunique certificates rather than barcode invariants.

### Why selective canceled-death correction is sufficient

The following theorem establishes **existence and uniqueness of the correction within the canceled-death subspace**, assuming the implementation performs the stated elementary contractions. It does not claim uniqueness among all possible original-grid fillings or prove anything about the particular spatial geometry of a chosen certificate.

**Elementary cancellation.** Work over `F₂`. Suppose the current filtered chain complex contains a pair of basis generators `a` (dimension `q`) and `b` (dimension `q+1`), with identical filtration value, such that

\[
\partial b=a+r,\qquad [a]r=0.
\]

All other generators are denoted `c`. Let the new complex be freely generated by those other generators, with `∂' = p ∂ i`. Define maps on basis generators by

\[
\begin{aligned}
p(a)&=r,&p(b)&=0,&p(c)&=c,\\
i(c)&=c+[a](\partial c)b,\\
h(a)&=b,&h(b)&=0,&h(c)&=0.
\end{aligned}
\]

Here `[a](x)` is the coefficient of `a` in `x`; in `p(a)=r`, `r` is regarded as a chain in the remaining generators. Direct substitution using `∂²=0` verifies

\[
pi=\mathrm{id},\qquad \partial i=i\partial',\qquad p\partial=\partial'p,
\qquad \mathrm{id}-ip=\partial h+h\partial.
\]

All maps are filtered. In particular, the homotopy sends `a` to `b` at exactly the same filtration value, and `i(c)` can add `b` only if `a` occurs in `∂c`, which forces `f(c) ≥ f(b)`.

**Composed contraction.** Perform the cancellations successively in their deterministic order. Composition gives a filtered projection `P` from the original complex `C` to the final compressed complex `C'`, inclusion `I`, and degree-`+1` homotopy `H` satisfying

\[
PI=\mathrm{id}_{C'},\qquad
\mathrm{id}_C-IP=\partial H+H\partial.
\]

More explicitly, for stepwise maps `p_j, i_j, h_j`, one can take

\[
H=h_1+i_1h_2p_1+i_1i_2h_3p_2p_1+\cdots.
\]

Every nonzero term of `H` is a canceled **death** generator lifted back through earlier inclusions. Each earlier inclusion adds only earlier canceled death generators, so the output of `H` in dimension `q+1` lies entirely in the span `B_{q+1}` of original canceled death cells of that dimension. This remains true after restricting to any sublevel `C_t`: only canceled pairs with filtration value at most `t` occur.

**Selective-correction theorem.** Let `z` be an original `q`-cycle supported by `C_t`; it may be the sum of a dying PPG cycle and *all* its outgoing target cycles. Suppose the compressed cycle `z' = Pz` is filled by a compressed `(q+1)`-chain `A'` at `t`, so `∂'A'=z'`. Define the linear section `s` by replacing each compressed surviving generator with its original cubical-cell ID; it satisfies `Ps=id`, but need not be a chain map. Put

\[
\varepsilon=z+\partial s(A').
\]

Then `∂ε=0` and `Pε = z' + ∂'A' = 0`. The contraction identity yields

\[
\boxed{\varepsilon=\partial H\varepsilon.}
\]

Consequently, `T=Hε` belongs to `B_{q+1}∩C_t` and satisfies `∂T=ε`. Thus

\[
\boxed{\partial\big(s(A')+T\big)=z.}
\]

No arbitrary original `(q+1)`-cell columns are required to **solve the correction**: canceled death columns suffice. Furthermore, their boundaries are linearly independent. Indeed, take any nonzero combination of canceled deaths and choose its *latest canceled* generator `b_j`. Projecting through the first `j−1` cancellations annihilates all earlier death terms but leaves `b_j`; its boundary is still nonzero because the current differential has pivot `a_j` with coefficient one. Therefore the original combination cannot have zero boundary. It follows that **`T` is unique in the canceled-death subspace**, although other fillings can differ by arbitrary higher-dimensional cycles.

**Strict-predeath test.** Because `P` and `I` are filtered chain homotopy inverses, they induce isomorphisms on each sublevel homology group. For a cycle supported below `t`, it is a boundary using original cells strictly below `t` *if and only if* its compressed image is such a boundary. Hence the pair-trace implementation can verify a genuine event-level death by reducing `z'` against compressed boundary columns born strictly before `t`, rather than constructing a full original-grid boundary matrix.

The proof assumes each canceled pair genuinely has a unit pivot in its **current contracted differential**, not just in the raw original boundary, and that the compressed reduction/PPG event relation and cycle projections are correct. The current tests verify these identities on finite examples; a complete mechanized invariant check of the streaming implementation and Rust–Python cross-language parity remain separate validation work. This is a chain-level algebraic theorem, not a claim that individual PPG edges independently bound or that the chosen geometric filling is physically optimal.

Tests cover the known pair-of-pants event with two outgoing edges, a 3D cavity, random 2D/3D images, and exhaustive binary 2×3 and 2×2×2 arrays. The small-test implementation is intended as a reference oracle for a later sparse, selective witness solver.

## Compact event-level graph summaries

A PPG can be very large. Event statistics can be computed while edges are written directly to CSV rather than retained in Python lists:

```python
from persistence_pairing_graph import compute_ppg

graph = compute_ppg(
    image,
    algorithm="stream-block",
    collect_event_summary=True,
    collect_degrees=True,
    edge_output_path="edges.csv",
)
graph.write_event_summaries("events.csv")

print(graph.edge_count)
print(graph.event_summaries[0])
print(graph.in_degrees, graph.out_degrees)
```

The summary records **homological dimension, death intensity, number of classes dying, and number of outgoing edges** for every death event, including events with no outgoing edges. The node degree arrays are optional (`uint64`, two arrays of length equal to the number of PPG nodes). Summary collection still computes each transition matrix; it avoids storing the final graph edges, not the transient event algebra. `graph.to_networkx()` requires retained edges.

## Independent Betti curves (no persistent matrix reduction)

For finite cubical complexes formed as unions of closed pixels or voxels with nonperiodic boundaries:

- foreground 8-connectivity (2D) or 26-connectivity (3D) gives `β₀`;
- background 4-connectivity (2D) or 6-connectivity (3D), with the exterior identified, gives the number of bounded complement components (`β₁` in 2D, `β₂` in 3D);
- counting cells by filtration and dimension gives Euler characteristic `χ`;
- in 3D, `β₁ = β₀ + β₂ − χ`.

This is a fast, independent topology diagnostic. It does **not** compute persistence birth/death pairings, representative cycles, or PPG edges.

```python
from persistence_pairing_graph import betti_curves, validate_ppg_betti

curves = betti_curves(image, accelerator="numba")
curves.to_csv("betti_curves.csv")

# Optional independent correctness check on an existing graph's barcode:
validate_ppg_betti(image, graph, accelerator="numba")
```

CLI:

```bash
ppg-betti volume.tif --accelerator numba --output betti_curves.csv
```

Curves are evaluated at each distinct image intensity. The independent calculation supports `uint8` and `uint16` 2D/3D arrays, with an explicit voxel-count memory guard.

## Original-grid representative cycles and filtered lifting

For smaller images, you can recover exact canonical homology cycles using the **uncompressed** `lazy` reduction's transformation-operation log:

```python
from persistence_pairing_graph import representative_cycles

cycles = representative_cycles(image, dimensions=(1, 2), max_features=10)
for cycle in cycles:
    print(cycle.dimension, cycle.birth_cell, cycle.death_cell)
    print(cycle.doubled_grid_coordinates)  # one row per cell of the cycle
    print(cycle.cell_centers)               # grid coordinates divided by two
```

Each chain `V_birth` is verified to have zero cellular boundary over F₂. The original cubical cells—not just a representative point—are returned. The chosen canonical chain need not be the smallest or geometrically shortest cycle. This routine has conservative memory and chain-size limits. The streaming backend can now reconstruct filtered representative **cycles**, too, using an optional disk-backed record of the equal-filtration cancellation pairs. This is not a full chain contraction map `(i, p, h)` for arbitrary chains, and lifted cycles need not equal the reference reducer's canonical original-grid representatives.

```python
from persistence_pairing_graph import representative_cycles_streaming

cycles = representative_cycles_streaming(
    image, dimensions=(1, 2), max_features=10,
    reduction_storage="packed", max_basis_columns=200_000,
    max_basis_entries=5_000_000, max_chain_terms=500_000,
    temp_dir="/scratch",  # optional; scratch files cleaned up automatically
)
for cycle in cycles:
    print(cycle.dimension, cycle.birth_value, cycle.death_value)
    print(cycle.original_grid_indices)     # every cell of the actual cycle
    print(cycle.doubled_grid_coordinates)  # spatial positions, one row per cell
```

The streaming contraction logs the original-grid *death cell*, dimension, and scalar value of each zero-length canceled pair (eight bytes per pair on scratch disk). For a compressed cycle `z'`, let `s` be the chain of its surviving original-grid generators. The reconstruction solves `∂x = ∂s` over `F₂` using the original boundaries of canceled death cells with filtration at most the feature's birth time. Then `z = s + x` satisfies `∂z = 0`, uses only cubical cells present at the birth time, and projects back to `z'` under the contraction. Each output is independently boundary-verified.

The same procedure can export compact original-cell supports through the command line:

```bash
ppg-lift volume.tif --dimensions 1 2 --max-features 10 \
    --output lifted_cycles.npz
```

The NPZ stores `support_offsets` and concatenated `original_grid_cell_ids`.
Feature `i` occupies `original_grid_cell_ids[support_offsets[i]:support_offsets[i+1]]`.
It also stores filtration endpoints and the input shape, so doubled-grid cell
coordinates can be reconstructed with `np.unravel_index`.

The sparse correction solve is **on-demand** and deliberately memory-limited. Its default safeguards are 200,000 cancellation-basis columns, five million stored basis entries, and 500,000 cells per returned representative. For more difficult images, adjust limits explicitly or analyze a crop. The optional trace increases temporary disk usage and recovery cost; ordinary `compute_ppg` runs do not record it. A complete inclusion/projection/chain-homotopy representation for *arbitrary* chains remains future work.

The streaming graph can optionally retain original grid indices for compressed generators:

```python
graph = compute_ppg(image, algorithm="stream-block", retain_cell_provenance=True)
print(graph.cell_coordinates(graph.nodes[0].birth_cell, image.shape))
```

These are the *birth generator's coordinates*, not the coordinates of a full homology cycle.

## Interpret PPG edges using original-grid geometry

Each nonzero PPG edge is an **algebraic transition** computed by the same
canonical reduction used for the persistence graph. To investigate where its
source and target classes occur in the original image, lift their birth-time
representative cycles into the original cubical grid:

```python
from persistence_pairing_graph import explain_ppg_edges

report = explain_ppg_edges(
    image, dimensions=(1, 2), max_edges=10,
    spacing=(1.0, 1.0, 1.0),  # omit for 2D or use two spacings
    max_geometry_pairs=1_000_000,
)
for item in report.edges:
    print(item.edge_index, item.edge.from_node, item.edge.to_node)
    print(item.event_value, item.shared_cell_count, item.min_distance)
    print(item.source.doubled_grid_coordinates)
    print(item.target.doubled_grid_coordinates)
    print(item.source_bounds, item.target_bounds)

report.write_csv("edge_geometry.csv")
report.write_npz("edge_geometry.npz")
```

The CLI exports a human-readable edge table plus a compact, non-pickle NPZ
with one copy of each unique source/target cycle support:

```bash
ppg-explain volume.tif --dimensions 1 2 --max-edges 10 \
    --spacing 1.0 1.0 1.0 --output-prefix edge_geometry
```

In the NPZ, `node_ids` gives the graph node identifiers for the stored cycles.
The original doubled-grid cell IDs for cycle `i` occupy
`original_grid_cell_ids[support_offsets[i]:support_offsets[i+1]]`.
Each edge is represented by `edge_indices`, `from_node`, `to_node`, and its
filtration-level `transition_values`. Convert cell IDs to doubled-grid
coordinates using `np.unravel_index(ids, doubled_grid_shape)`. The original
voxel-centered coordinates are the doubled-grid coordinates divided by two.

**What the geometric measurements mean:** `shared_cell_count` counts identical
cubical cells appearing in both lifted chains. `minimum_cell_distance` is the
exact minimum distance between the *closed geometric supports* of those cells,
in physical units determined by `spacing`. Distinct cubical cells can touch at
a shared face or vertex, giving zero distance even with no shared cell IDs.
When the pairwise support product exceeds `max_geometry_pairs`, the distance
is omitted (Python `None`, CSV blank, NPZ `NaN`) rather than approximated.
A nonzero graph edge does **not** imply spatial intersection or adjacency;
it records a basis-dependent algebraic transition. Likewise, the reported
lifts represent classes at their **birth filtration values**, not optimized
shortest cycles or a geometric witness of their death-time transition.

For reproducibility, `edge_indices` selects exact zero-based positions in the
full canonical edge stream, even if dimensions are filtered:

```python
selected = explain_ppg_edges(image, edge_indices=[2, 7], max_edges=2)
```

This API reuses one streaming contraction and one canonical sparse reduction
for both edge selection and targeted lifting. As with `ppg-lift`, the optional
disk-backed cancellation trace is removed after completion; sparse lifting
has explicit memory and work caps. For large volumes, investigate a few
selected edges at a time.

## Multiscale and perturbation experiments

Explicit min/max spatial pooling supports exploratory multiscale Betti summaries:

```python
from persistence_pairing_graph import multiscale_betti_curves

profiles = multiscale_betti_curves(image, factors=(1, 2, 4),
                                   reducer="min", accelerator="numba")
for profile in profiles:
    print(profile.factor, profile.shape, profile.curves.beta1)
```

This computes topology on **different coarsened images**, not a feature-level vineyard or a guaranteed persistence-preserving correspondence across resolutions.

For exploratory graph-level sensitivity to bounded integer noise:

```python
from persistence_pairing_graph import ppg_perturbation_probe

report = ppg_perturbation_probe(image, repeats=5, amplitude=1, seed=23)
report.to_csv("perturbation_summary.csv")
```

Sample zero is unmodified; subsequent samples receive independent clipped intensity perturbations. Reports contain total node and edge counts, dimension-specific counts, and finite total persistence. **No feature matching is assumed**, so these summaries do not establish stability of individual PPG edges or statistical confidence intervals.

## Validation and benchmarking

```bash
python -m pytest -q
python benchmarks/bench_new_features.py --size 40 --repeat 3 --output benchmark.json
python benchmarks/profile_compiled.py --case random --size 40 --accelerator numba
python benchmarks/run_structured_suite.py --sizes 24 48 --output structured.json
```

The tests cover materialized versus streaming compressed boundaries, deterministic canonical operation logs, graph edges, simultaneous-death events, in-memory versus disk-backed metadata, raw and delta ordering, compiled counting scatter, direct CSV export, analytic cycles, filtered streaming representative lifts, independent Betti curves versus persistence barcodes, multiscale pooling, perturbation reproducibility, and small binary images exhaustively.

To perform true cross-language validation, install Rust/Cargo and use a checkout of the reference Rust repository:

```bash
python tests/validate_rust_suite.py \
    --rust-repo /path/to/persistence_pairing_graph \
    --algorithms stream-block block lazy --uint16 \
    --json rust_python_results.json
```

This cross-language execution has **not** yet been performed in the current execution environment, which lacks Rust tooling. A successful Python-only regression suite is not proof of Rust parity.

## Scientific limitations

Persistence barcodes are invariants of the filtered complex; selected reduction-basis PPG edges are not. Alternative boundary reductions, cohomological shortcuts, union-find pairing, and arbitrary clearing must **not** replace the canonical PPG reducer without recovering the necessary basis maps. Some expensive proposals remain research directions: complete compiled cancellation/finalization, distributed value-block contraction with chain-map recovery, fully out-of-core end-to-end 256³ processing, complete arbitrary-chain homotopy maps through streaming contraction, persistent cup products, and principled individual-edge robustness under tied or perturbed filtrations.

The project is released under the MIT License. For research context and citation information, see the original [persistence-pairing graph Rust repository](https://github.com/jhnrckmnznrs/persistence_pairing_graph).

## Reusable analysis and selective transition witnesses (experimental)

For several selected PPG transitions from one image, use a single analysis
session rather than recompressing and re-reducing the image for every witness:

```python
import numpy as np
from persistence_pairing_graph import PPGAnalysis

image = np.zeros((14, 14, 14), dtype=np.uint8)
image[7, 7, 7] = 90
with PPGAnalysis(image, max_basis_columns=100_000) as session:
    candidates = [i for i, node in enumerate(session.nodes)
                  if node.dimension == 2 and not node.essential]
    if candidates:
        certificate = session.witness(candidates[0], solver="component")
        print(certificate.filling_cell_count, session.last_component_stats)
```

`PPGAnalysis` owns one compressed complex, canonical reduction, disk-backed
cancellation trace, and event-specific target cache. It also caches recovered
birth-cycle representatives and incrementally builds their cancellation bases
in increasing birth-value order. Asking for an earlier uncached birth value
rewinds that degree's basis; certificates already returned remain valid.
Use it as a context manager or call `close()` to remove temporary files. A bounded LRU cache of up to four complete witnesses is enabled by default (`max_cached_witnesses`), and may be disabled with zero.

`solver="component"` constructs the correction only from canceled death
cells in the **incidence-connected components touched by the residual**.
This method still starts with the complete *list of eligible canceled deaths*
from the mapped trace and creates a sorted search index. It avoids creating a
full original-grid boundary basis, but is **not** fully out of core. The
alternative `solver="pair-trace"` builds the full eligible cancellation basis
for comparison. Both return exactly verified full death-event certificates,
not purported certificates for individual PPG edges. `max_component_faces`,
`max_basis_columns`, `max_basis_entries`, `max_chain_terms` and `max_work` bound
resource consumption. Exceeding a limit raises `MemoryError`, never a partial
or silently approximate filling.

### Why incidence-component restriction is exact

Fix a homological degree `q` and filtration value `t`. Let `D` be the
original cubical boundary map restricted to **canceled `(q+1)`-death columns
born at or before `t`**, with all original `q`-cells as rows. Construct the
bipartite incidence graph whose vertices are these columns and the `q`-rows;
a column is adjacent to a row exactly when its boundary coefficient is `1`.

Let `epsilon` be the correction target, and let `S` be the union of connected
components containing a nonzero row of `epsilon`. Reordering rows and columns
by connected component makes `D` block diagonal. Consequently, if
`D x = epsilon` has a solution, setting all coefficients outside `S` to zero
still solves the equation. The canceled-death correction theorem above
establishes existence in the full canceled-death span. Hence the **restricted
incidence components suffice**. Moreover, the canceled-death columns are
independent under the elementary-contraction assumptions: the selected
correction is the unique solution in this span, including the same solution
as the full canceled-pair method. The component search uses the doubled-grid
geometry to enumerate the at most `2(n-q)` cubical cofaces of a `q`-cell; no
geometric overlap heuristic or approximate proximity threshold enters the
algorithm.

This restriction has an important worst case. One incidence component may
span nearly every cancellation. Selective recovery then offers little benefit
and may have extra traversal and indexing overhead. Measurements should report
component sizes, not just wall-clock time. The method does **not** implement
an explicitly stored arbitrary-chain contraction `(i,p,h)` or a minimal
geometric filling. It returns exactly checked witnesses and retains the
filtration-respecting `R=DV` reduction convention.

Use the benchmark below to compare first and repeated witness requests:

```bash
python benchmarks/benchmark_selected_witnesses.py --size 10 --repeats 3 \
    --output witness_benchmark.json
```

The repository's `tests/validate_rust_suite.py` can perform Rust-versus-Python
CSV comparisons when Rust tooling and the original repository are available.
That external cross-language validation has not been executed here.
