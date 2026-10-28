# Local filling certificates for selected PPG transitions

This experimental extension implements a **bounded exact oracle** for the
question: does the complete selected death-event relation bound inside a
prescribed set of original cubical cells?

It complements the existing `transition_filling_witness` and
`SelectiveInclusionReplay` implementations. It does **not** compute a
minimal-area or otherwise optimal representative, and it is not a
large-volume persistence reduction.

## Mathematics

Let (c \in Z_q(K_t;\mathbb F_2)) denote the **complete** relation
formed from a selected PPG source cycle and all of its target cycles.
For a proposed set (T) of eligible ((q+1))-cells, form the restricted
boundary matrix (B_T). The oracle solves

```
B_T x = c   over GF(2)
```

If solvable, it returns the original-grid filling (x), independently
checked against the cellular boundary operator.

Otherwise, it returns a (q)-cochain (\lambda) satisfying

```
lambda^T B_T = 0       and       lambda^T c = 1.
```

The cochain is an exact *dual obstruction*: no chain supported in the
proposed cell set can fill the relation. It says nothing about the
existence of fillings outside that set.

The solver uses local row indexing and bitset Gaussian elimination.
Its work and memory have explicit input/resource caps but may still
grow rapidly with the number of allowed cells. It returns **one**
solution, not the sparsest or shortest solution.

## Usage

```python
import numpy as np
from persistence_pairing_graph import (
    transition_filling_witness,
    certify_local_filling,
    certify_transition_filling_locality,
)

# image is a 2D or 3D uint8/uint16 scalar array.
w = transition_filling_witness(image, source_node=2)

# The existing exact witness must remain feasible when its own
# original-grid filling cells are allowed.
check = certify_local_filling(
    w.relation_support,
    w.filling_grid_indices,
    image.shape,
    dimension=w.dimension,
)
assert check.feasible

# Investigate a proposed region, with inclusive bounds in
# *doubled-grid cell-center coordinates*.
local = certify_transition_filling_locality(
    image,
    w,
    lower=(0, 0),
    upper=(2 * image.shape[0], 2 * image.shape[1]),
)

if local.feasible:
    print("Verified filling:", local.filling_cells)
else:
    print("Verified dual obstruction:", local.obstruction_cells)
```

For 3D arrays, provide three-coordinate bounds. The convenience box
enumerator uses the repository's lower-star cubical convention and includes
only cells with filtration value `<= w.event_value`. For other filtration
conventions, construct an explicit candidate list and call
`certify_local_filling` directly.

A region restricts **filling cells**, not necessarily the cycle's
representative cells. A successful certificate is a literal original-grid
chain identity, not just an abstract homological equivalence.

## Correctness and scope

- This extension uses the complete PPG death relation, **not** any one
  outgoing graph edge in isolation.
- The algorithm works over (\mathbb F_2); orientations and coefficients
  over other fields are not implemented.
- It can establish the sufficiency or insufficiency of an explicitly
  specified region; it cannot prove that the result is physically unique.
- Dual obstructions are defined relative to the candidate cells. They
  need not be cocycles in the entire ambient cubical complex.
- The method does not provide unconditional spatial stability or
  complexity bounds for arbitrary filtrations.
- For large-scale reconstructions, prefer the existing selective
  transition-witness backend and use this oracle only on bounded regions.

The new tests compare exact feasibility with independent exhaustive
subset enumeration on small 2D and 3D grids and verify the dual pairing
equations. The pair-of-pants example also checks compatibility with the
existing original-grid PPG filling witness.
