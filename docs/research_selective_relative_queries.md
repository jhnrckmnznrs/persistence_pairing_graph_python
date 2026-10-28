# Selective relative queries and 3D dependency closures — research audit

**Status:** exact bounded reference implementation and reproducible controlled
synthetic diagnostics. The data below are GitHub Actions measurements on one
runner; they are **not** benchmarks on actual porous-media or micro-CT files.

## 1. Exact old-incidence restriction

At a filtration value t, set A=K_{<t}, X=K_{<=t}; for a selected old
q-cycle c, and a prescribed set T of new (q+1)-cells, solve over F2:

```
d_A a + F_T z = c,       D_T z = 0.
```

Let G_A be the bipartite incidence graph connecting old q-faces to old
(q+1)-cells whenever the corresponding original cubical boundary coefficient
is nonzero. Define S to be the union of supp(c) and the boundaries of all
permitted new (q+1)-cells, including their **new** q-faces.

**Proposition (component completeness).** Let G_A[S] be the union of old
incidence components meeting S. There exists a solution (a,z) to the above
equations if and only if there exists one with a supported on old columns in
G_A[S].

*Proof.* The old boundary matrix is block diagonal across connected
components of G_A, while all new columns have their nonzero rows in S.
In a component disjoint from S, the equation has zero right-hand side
and no contribution from permitted new columns. Setting the coefficients
of a on that component to zero preserves the equation. The converse is
immediate. The new q-faces never have old cofaces because A is a
subcomplex. Thus processing the union of seeded old components is exact.

**Dual correctness.** For any infeasible restricted solve, the returned
cochain lambda has lambda(c)=1 and annihilates every selected old column
and every permitted new column. Because all unvisited old columns lie in
incidence components disjoint from the selected rows, lambda also
annihilates **all** old columns. This certifies genuine infeasibility
for the prescribed T, not merely failure of a geometric heuristic.

**Bounded work.** In a fixed-dimensional cubical complex, an old q-cell
has at most 2(d-q) cofaces of dimension q+1; each (q+1)-cell has 2(q+1)
boundary faces. Discovery therefore takes O(number of visited incidence
vertices and edges) operations, plus on-demand filtration tests. The
subsequent GF(2) elimination remains potentially dense and worst-case
superlinear. No universal sublinear or optimal-filling claim is made.

**Negative example.** In a full cubical box, the q=1 old edge-square
incidence graph can be connected throughout the box. Even a cycle that
is the boundary of a *single old square* then causes full component
exploration, despite having a constant-sized filling. Component
connectivity alone cannot certify small dependency size.

## 2. A fast, one-sided positive certificate

When the existing PPG implementation has already returned a verified
original-grid filling Gamma of c, split its cells at t into an old
correction a and new cells z. If all new cells lie in the user's
proposed region T, Gamma is an immediate exact certificate.

This fast path checks the original cellular boundary, dimensions and
filtration values and returns `old_solver='witness'`, with no old
incidence search or elimination. The sentinel `boundary_rank=-1`
means no matrix rank was calculated. If Gamma uses a new cell outside
T, this does **not** prove infeasibility: another permitted filling
could exist. The algorithm then falls back to the complete incidence
solver.

Obtaining the existing PPG filling is not free. This only avoids an
**additional** relative-locality computation when the witness is
already available.

## 3. Observed reverse-replay dependency closures

The benchmark samples up to eight compressed surviving generators per
degree 1 and 2, uniformly spaced in compressed filtration order.
It records the **distinct cancellation indices in queried influence
sets**, not the entire static Morse graph and not all work required to
compute the initial reduction.

| n | Pattern | Canceled pairs | Queries | Median reached fraction | Maximum reached fraction | Max observed depth |
|---|---|---:|---:|---:|---:|---:|
| 8 | Noise | 2,348 | 16 | 0.4685% | 0.6814% | 12 |
| 16 | Noise | 16,919 | 16 | 0.0680% | 0.2837% | 12 |
| 24 | Noise | 54,941 | 16 | 0.0173% | 0.0710% | 7 |
| 32 | Noise | 127,813 | 16 | 0.0074% | 0.0243% | 9 |
| 8 | Correlated | 2,425 | 16 | 0.5773% | 3.9175% | 19 |
| 16 | Correlated | 17,737 | 16 | 0.0930% | 0.4905% | 12 |
| 24 | Correlated | 57,935 | 16 | 0.0311% | 0.2589% | 20 |
| 32 | Correlated | 135,245 | 16 | 0.0118% | 0.0458% | 13 |
| 8 | Cavity | 2,455 | 1 | 3.2587% | 3.2587% | 10 |
| 16 | Cavity | 17,967 | 1 | 1.4137% | 1.4137% | 19 |
| 24 | Cavity | 58,823 | 1 | 1.0999% | 1.0999% | 31 |
| 32 | Cavity | 137,311 | 1 | 0.8564% | 0.8564% | 42 |

Random and correlated single-generator queries touch a small
*absolute* number of cancellation records (roughly ten to twenty by
the larger sizes). The single cavity query grows from approximately
80 records at n=8 to 1,176 at n=32, consistent with roughly
surface-area-like growth across these five tested resolutions.
Neither observation is an asymptotic guarantee.

The layered field has no suitable sampled positive-degree generator
at n=8, 12, or 16; the zero-query cases are **missing observations**,
not measured zero complexity.

### Actual PPG death-event samples

A separate benchmark uses `PPGAnalysis.replay_dependency_closure`
on up to two nonzero-persistence finite PPG death events per volume,
preferring H1/H2:

- n=5 correlated: one query reached 23/657 canceled pairs;
- n=7 correlated: one query reached 61/1,673 (3.65%);
- n=9 correlated: two selected queries reached 7 and 6 of 3,387;
- n=9 cavity H2 event: 53/3,428 (1.55%).

These are *small controlled examples* and selected events, not all
PPG events or a population-level estimate. The n=5/7/9 range is too
short to infer event-level scaling, and large important witnesses
may exhibit very different closures.

## 4. Incidence selection results

For one-square relative events with old boundary loops, the exact
incidence selection retained between 99.1% and 100% of the old
columns in n=5,7,9 noise/correlated cases. Most cavity/layer cases
had no suitable one-square event under this benchmark's fixed
thresholds and are explicitly **skipped**, not assigned zero cost.
Measured total solve times were similar in this range, with no
decisive performance advantage.

Thus the component restriction is an exact improvement on
disconnected inputs but does not solve the main global-connected
old-boundary bottleneck. It should remain a reference / fallback,
not be advertised as a scalable general PPG localization backend.

## 5. The next selective algebraic query (proposed, not implemented)

Let E_q be an evaluation map that sends old q-cycles to coordinates
in H_q(A) using a basis of dual cocycles. Given the relative
new-cell boundary blocks D_T and F_T, compute a basis Z_T
of ker D_T. A permitted relative filling exists exactly when the
small (or at least restricted) coordinate system

```
    E_q F_T Z_T u = E_q c
```

is solvable over F2.

*Proof.* D_T Z_T u=0 ensures F_T Z_T u is an old q-cycle
by d^2=0. The difference c+F_T Z_T u is an old boundary
if and only if every old cohomology class evaluates to zero.
This is precisely the coordinate equation above.

Once u is found, reconstruct a concrete old correction from

```
    d_A a = c + F_T Z_T u.
```

This splits the query into (a) rank/feasibility in a potentially much
smaller cohomology coordinate space and (b) optional explicit
old-chain reconstruction. A negative coordinate result also
provides a dual obstruction after an appropriate cochain lift.

**Crucial limitation:** The current Python PPG solver does not expose
a reusable complete old-level dual cocycle coordinate evaluator and
a matching selective boundary preimage oracle. Building these
from the existing filtered compression, saved reduction data and
inclusion replay—without materializing dense global matrices—is the
next substantive research milestone, not something already achieved
by the incidence solver.

This approach must be compared with U-match factorization, efficient
image-persistence algorithms, and sparse reduction variants rather
than presented as novel homological algebra.

References:
- Bauer and Schmahl (2025), *Efficient Computation of Image Persistence*:
  https://doi.org/10.1007/s00454-025-00769-8
- Hang et al., *U-match factorization*:
  https://arxiv.org/abs/2108.08831
- Bauer et al. (2024), *Keeping it sparse*:
  https://doi.org/10.57717/cgt.v3i1.50


## 5b. Query-local forward projection (implemented on the research branch)

Each recorded equal-filtration elementary contraction has a birth generator a,
a death generator b and a tail r with d_j(b)=a+r over F2. Its chain
projection sends a to r, b to 0, and fixes the other current generators.
Composing these maps in the order of cancellation gives the projection from
the original cubical complex to its compressed chain complex.

`SelectiveProjectionReplay.project(original_grid_cells)` applies these
substitutions using a min-heap of only touched canceled generators; no
dense global projection matrix and no intermediate differential are formed.
`stop_before_value=t` applies only cancellation records entering
strictly before t, making it possible to project chains in the old complex
K_{<t} without canceling the event-level pairs.

Independent tests verify:
- agreement with explicit elementary cancellations on small 2D and 3D grids;
- exact chain-map identity d_M p(c)=p(d_C c);
- p i=id on selected compressed generators, with i computed by the
  separately implemented reverse replay;
- exact agreement of filtration-truncated projections.

This supplies the **projection** half of a selective cohomology-coordinate
query. It does not by itself supply the old-level cocycle basis E_q or the
selective old-boundary preimage needed to solve the full coordinate system.
The forward trace must remain available and its filtration levels must
respect the recorded cancellation order.


### Pre-death projection of complete PPG relations

The event-level benchmark also projects each **complete** selected death
relation through only the cancellations with value strictly below its death
time. These are meaningful chain queries, unlike the trivial single canceled
birth probes used for the separate projection microbenchmark. All tested
pre-death relations remained nonzero after projection, as required by their
nontrivial old homology class.

Examples: at n=5, the cavity H2 relation used 32 original cycle cells,
touched 31 cancellation records and projected to one surviving cell;
at n=9, the cavity H2 relation used 90 cycle cells, touched 89 records
and also projected to one surviving cell. One n=7 correlated H1 relation
used 24 cycle cells, touched 21 cancellations and projected to three cells.
These are small synthetic examples, **not** representative worst-case
complexity bounds. The tested relation projections read no nonzero stored
tail terms, so dense-tail cases still require a targeted stress test.

## 6. Reproducibility and scope

```sh
python -m pytest -q
python benchmarks/investigate_dependency_closure.py --sizes 8 12 16 24 32
python benchmarks/benchmark_selective_relative.py --sizes 5 7 9
python benchmarks/benchmark_event_closure_3d.py --sizes 5 7 9
python benchmarks/benchmark_projection_3d.py --sizes 8 16 24 32
```

Exact relative queries are validated against the independent full
enumeration and dual annihilation checks. Synthetic closure figures
are not mechanical, predictive, clinical, or real-material results.
No changes to Manuscript II's empirical claims should be made until
the repository is used on actual porous media and representative
micro-CT volumes with an agreed verification protocol.
