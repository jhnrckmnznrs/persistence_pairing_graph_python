# Relative filling locality and dependency-closure statistics

These are **experimental, exact, bounded** extensions to the existing PPG
witness and reverse-replay functionality. They do not replace the streaming
reduction and do not attempt optimal or minimum-support fillings.

## 1. Relative localization: old corrections remain unrestricted

For a death event at a distinct scalar value `t`, take
`A = K_{<t}` and `X = K_{<=t}`. Let `c` be a complete PPG
death-event cycle in `C_q(A; GF(2))`. Decompose the chain groups into
old and new cells. For a selected set `T` of new (q+1)-cells, we test

```
d_A a + F_T z = c
          D_T z = 0
```

where **all** old (q+1)-cells in A are permitted in a, but only
new (q+1)-cells in T are permitted in z. The old correction is NOT
restricted to the new-cell box.

`certify_relative_filling` explicitly enumerates the eligible old
cells and solves the combined original-grid boundary equation. Its
outputs independently verify both the literal boundary equation and the
relative-cycle condition for z.

If no solution exists, a dual cochain lambda is returned satisfying

```
lambda(c) = 1
lambda(d_A a) = 0 for every old (q+1)-cell a
lambda(d z) = 0 for every allowed new (q+1)-cell z.
```

This proves that **no permitted relative filling exists**. It does not
prove that the relation cannot be filled using new cells outside T.

### Example

```python
from persistence_pairing_graph import (
    transition_filling_witness,
    certify_transition_relative_locality,
)

w = transition_filling_witness(image, source_node=2)
result = certify_transition_relative_locality(
    image, w, lower=(2, 2), upper=(8, 8))
if result.feasible:
    print("old correction:", result.old_filling_cells)
    print("new cells inside box:", result.new_filling_cells)
else:
    print("dual obstruction:", result.obstruction_cells)
```

Bounds are inclusive doubled-grid coordinates. For 3D data use
three-coordinate bounds. Only the newly entering cells are box-constrained;
old correction cells can lie anywhere in `K_{<t}`.

For an explicitly specified permitted new set, use
`certify_relative_filling(image, relation_cells, allowed_new_cells,
dimension=q, event_value=t)`.

**Limitations:** These functions use the uint8/uint16 cubical scalar
filtration and simultaneous value-level events. They enumerate all old
(q+1)-cells and apply bounded GF(2) elimination; runtime and memory can
grow quickly. They do not optimize support, and the returned filling is
not canonical. A PPG row must be used in full, not an isolated graph edge.

## 2. Reverse replay: query-local dependency closure

`SelectiveInclusionReplay.lift(cells, collect_dependency_stats=True)`
provides `last_closure_stats`, separately from the existing
`last_stats`. This is opt-in to avoid collecting graph sets on routine
calls.

A **reachable record** is any cancellation index appearing in an
influence set requested during the selected lift, including previously
memoized influences. A directed **dependency arc** `j -> k` is counted
whenever replay of a nonzero cancellation `j` requests an influence
containing the earlier cancellation `k < j`.

Reported fields:

- `seed_cells`: uncanceled generator IDs supplied to the lift.
- `influence_columns_queried`: influence lookups from seeds and inserted deaths.
- `reachable_records`: distinct cancellation indices encountered.
- `dependency_arcs`: potential replay dependence arcs encountered.
- `replayed_records`: nonzero cancellations actually replayed.
- `maximum_observed_depth`: longest arc chain in the encountered DAG.
- `total_cancellations` and `reachable_fraction`.

Some reachable cancellations cancel mod 2 and are never replayed.
The statistics include only the dependency graph *encountered by a
particular query*. They do not bound the entire contraction graph;
constructing the contraction, owner index and initial persistence
reduction is excluded. Existing `ReplayStatistics` counts the actual
work for the current call (including cache-dependent visits); closure
counts remain defined across cold and warm caches.

Use `PPGAnalysis.replay_dependency_closure(source_node)` in an analysis
session with `retain_inclusion_trace=True` to inspect just the selected
death-event **filling** lift (not the cycle lifts or preprocessing).
The method recomputes this query, verifies the original-grid result
against the existing exact witness, and returns the closure report.

```python
from persistence_pairing_graph import PPGAnalysis

with PPGAnalysis(image, retain_inclusion_trace=True) as analysis:
    stats = analysis.replay_dependency_closure(source_node=2)
    print(stats.reachable_records, stats.replayed_records)
    print(stats.maximum_observed_depth, stats.reachable_fraction)
```

These quantities test the structural assumptions in Manuscript I. A
small observed closure is not a worst-case complexity theorem, and the
reported `maximum_observed_depth` is not the full gradient-path depth.
