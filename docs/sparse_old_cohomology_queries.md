# Sparse old-level cohomology and relative PPG certificate queries

Research implementation (GF(2), 2D/3D uint8/uint16 lower-star cubical filtrations).
This is an extension of the *existing* PPG reduction and streamed filtered
contraction; no new repository or manuscript is needed.

## 1. Old-level quotient coordinates

Fix a filtration value t and A=K_{<t}. Let p:C_*(A)->M_*(A)
be the exact filtered projection encoded by the equal-value contraction
trace. Let R_j=d_M V_j be the nonzero reduced boundary columns of the
*compressed* persistence reduction whose death values satisfy value(j)<t.

The R_j of degree q form a basis for B_q(M(A)). Reduce p(c) by these
columns using their distinct highest pivots. The remaining vector defines

    E_{q,t}(c) := NF_{R^{<t}}(p(c)).

This map is linear on C_q(A), and its coordinate functionals annihilate
every old q-boundary. Accordingly, they are an exact (generally REDUNDANT)
cocycle-coordinate system; they are not promised to be a minimal selected
basis of H^q(A). For any OLD q-cycle c,

    E_{q,t}(c) = 0  if and only if  [c]=0 in H_q(A).

The index is built once from the already-reduced compressed complex.
The oracle uses a packed numeric pivot array, a numeric sorted map from
original-grid survivor IDs to compressed indices, and one shared sparse
projection/inclusion owner index. It does not enumerate all old original-grid
(q+1)-cells for each query. Resource caps still apply; sparse columns or
their expansions can be dense in the worst case.

The method relies on storing canonical reduced boundary columns and a
complete operation log. In the current PPGAnalysis, these are retained.
Other reduction modes may require additional metadata.

## 2. Restricted relative filling, separated into decision and reconstruction

Let B be the union of newly arriving cells at t. Restrict the allowed new
(q+1)-cells to T, and split their original cubical boundary into blocks

    d_X|_T = ( F_T ; D_T ),

where the first block has old q-face rows and the second has new q-face
rows. Choose a basis Z_T of ker D_T. A complete selected PPG death
relation c lies in C_q(A) and is a cycle.

There exists a permitted full original-grid filling c=d(a+z),
with a in C_{q+1}(A) and z supported on T, precisely when

    E_{q,t} F_T Z_T u = E_{q,t} c

is solvable. This follows from d²=0, filtered chain equivalence,
and the definition of E. The linear algebra over candidate new cells is
bounded by the number of candidates and their relative boundary rows,
not the number of old original-grid filling cells.

On failure, the solver returns a linear functional of the quotient
coordinates that evaluates to 1 on c and to 0 on every allowed relative
cycle boundary. This is an EXACT *lazy cocycle* obstruction:
`evaluate_cocycle` can evaluate it on any old q-chain.
The dense original-grid cochain itself is not materialized.

On success, `reconstruct=False` computes only the selected new chain z
and a mathematical feasibility certificate; the old correction a is NOT
returned. The result field `old_filling_reconstructed` distinguishes this
from a complete original-grid witness.

With `reconstruct=True`, the solver additionally finds a concrete old
chain a and checks the literal equality d(a+z)=c.

## 3. Sparse exact old-chain preimages by chain homotopy

For an old cycle b with E_{q,t}(b)=0, use reduced old compressed boundary
columns and their canonical V columns to construct y in M(A) satisfying

    d_M y = p(b).

Use selective inclusion replay to obtain i(y) in the original grid.

An elementary cancellation d_j(b_j)=a_j+r_j has a projection p_j
and inclusion i_j. It also has a chain homotopy h_j with h_j(a_j)=b_j,
h_j(other)=0 and

    id - i_j p_j = d h_j + h_j d.

Compose the cancellations in filtration order. On a cycle b,

    b + i p(b) = d H(b).

The new implementation records which canceled births a_j were activated
when projecting b. It computes H(b) by XOR-ing only the partial inclusions
of their paired death cells b_j. Therefore an exact original-grid
preimage is

    a = i(y) + H(b).

Every returned a is verified by the original cellular boundary operation
and by its strict old-sublevel filtration membership.

This is output/dependency-sensitive rather than a full original-grid
boundary enumeration. It can still be expensive when the queried
chain or its dependency closure is large; it does not minimize support.

For independent validation, `boundary_preimage(...,
correction_method='incidence')` retains the older exact canceled-death
incidence correction. The default is `correction_method='homotopy'`.

## 4. Example

```python
from persistence_pairing_graph import (
    PPGAnalysis, SparseOldLevelOracle,
)

with PPGAnalysis(image, retain_inclusion_trace=True) as analysis:
    w = analysis.witness(source_node, solver="replay")
    oracle = SparseOldLevelOracle(analysis)

    # The complete selected PPG relation is not an old boundary:
    old = oracle.coordinates(
        w.relation_support, dimension=w.dimension,
        event_value=w.event_value,
    )
    assert not old.is_boundary

    # Use candidate event-level cells from a verified PPG filling.
    from persistence_pairing_graph._transition_witness import _original_cell_birth
    new = [
        int(v) for v in w.filling_grid_indices
        if _original_cell_birth(image, int(v), analysis.grid_shape)
           == w.event_value
    ]

    decision = oracle.restricted_relative(
        w.relation_support, new,
        dimension=w.dimension, event_value=w.event_value,
        reconstruct=False,
    )

    complete = oracle.restricted_relative(
        w.relation_support, new,
        dimension=w.dimension, event_value=w.event_value,
        reconstruct=True,
    )
    print(decision.feasible, complete.old_filling_reconstructed)
```

The allowed new cells can instead come from a user-selected region.
This API is independent of a particular box representation; callers
should validate any proposed region before generating its permitted cells.

## 5. Verification and limitations

- The oracle and full original-grid reference give the same old-boundary
  decisions on tested random small 2D and 3D complexes, including H0/H1/H2.
- Bounded random relative queries agree with full original-cell Gaussian
  elimination in both feasibility and exact reconstructed boundaries.
- Complete PPG death-event relations in the pair-of-pants example pass;
  they fail at the pre-death old level and become boundaries after the event.
- The filtered chain-homotopy identity is checked independently of
  persistence reduction on random nested cubical examples.
- The lazy dual evaluation annihilates all old boundaries in tested samples.
- Nontrivial synthetic partial-inclusion cases test the cancellation trace.
- No proof of unconditional small dependency closure is claimed.
- The current test suite does NOT establish efficiency on full bone
  micro-CT volumes or the 119 porous-media volumes; those are future data
  validation questions.
- The coordinate array is redundant, and its values depend on the fixed
  compressed reduction and filtration ordering.
- Initial compression and persistence reduction may dominate one-shot use,
  so report their cost separately from reused-oracle query times.
- A full filling may be large even if the feasibility decision is cheap;
  do not conflate decision and construction complexity.

## 6. Reproducible performance audit

```sh
python -m pytest -q
python benchmarks/benchmark_old_cohomology.py --sizes 5 7 9 12 16 --events 2
```

The synthetic benchmark compares a cached compressed cohomology feasibility
query, full sparse-original-grid reconstruction, and a bounded reference
that enumerates all old original-grid filling cells. It reports the shared
PPG setup and oracle index time separately. These are single-run
measurements and should not be described as stable speedups.

### Preliminary measurements from GitHub Actions (28 October 2026)

These are **individual** selected-event timings in milliseconds after
the shared PPG reduction has been computed. A coordinate query answers
feasibility; a complete query also reconstructs and verifies the entire
original-grid chain. Times fluctuate and do not represent medians across
independent volumes.

| Synthetic 3D event | Feasibility only (ms) | Complete filling (ms) | Full original old-cell reference (ms) |
|---|---:|---:|---:|
| 9³ noise H1 | 0.58 | 0.90 | 22.35 |
| 12³ noise H1 | 0.36 | 0.99 | 29.80 |
| 16³ noise H1 | 0.31 | 0.47 | 65.79 |
| 16³ correlated H1 | 0.74 | 1.17 | 72.16 |
| 16³ cavity H2 | 7.21 | 12.73 | 46.02 |
| 24³ cavity H2 | 40.46 | 75.65 | Not run |
| 32³ noise H1 | 0.76 | 1.14 | Not run |
| 32³ cavity H2 | 91.83 | 171.79 | Not run |

One-time PPG setup at 32³ took 2.81 s for the cavity and 4.19 s
for the noise field on this runner. The oracle's compressed numerical
index construction took approximately 2.5 ms and 8.1 ms, respectively.
These costs must be included in comparisons for one-shot users.

The positive test uses new cells extracted from an existing verified
PPG filling. The negative test removes all permitted new cells; it is
expected to be infeasible for a genuine PPG death event. This does not
yet quantify effectiveness across arbitrary user-selected regions or
hard instances with many competing new relative cycles. Consequently,
the results demonstrate a promising precise implementation, **not**
unconditional complexity improvements or general predictive impact.

### Literature context

This use of reduction-based coordinates and duality is a specialization
of existing persistent homology algebra rather than an independent claim
to invent sparse cohomological queries.

- Hang, Giusti, Ziegelmeier, Henselman-Petrusek,
  *U-match factorization: sparse homological algebra, lazy cycle
  representatives, and dualities in persistent (co)homology*,
  https://arxiv.org/abs/2108.08831
- Bauer and Schmahl (2025), *Efficient Computation of Image Persistence*,
  https://doi.org/10.1007/s00454-025-00769-8
- Lentz, Henselman-Petrusek, Ziegelmeier (2026),
  *A U-match Algorithm for Persistent Relative Homology*,
  https://arxiv.org/abs/2602.03163
