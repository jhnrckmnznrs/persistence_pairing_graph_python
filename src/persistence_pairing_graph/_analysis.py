"""Reusable, resource-bounded PPG analysis and exact selected event witnesses.

A session owns one streaming compressed complex, one canonical reduction and
one cancellation trace. Event rows and lifted representatives are cached so
multiple certificates do not reconstruct the global filtration each time.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from math import prod

import numpy as np

from ._graph import build_nodes, iter_edges
from ._lift import (_BoundarySolver, _compressed_v_column, boundary_old_chain,
                    LiftedCycleRepresentative)
from ._local_pairs import ComponentSolveStats, solve_component_correction
from ._reduction import reduce_boundary
from ._reverse_replay import SelectiveInclusionReplay
from ._streaming import compress_discrete_tensor_streaming
from ._transition_witness import (TransitionFillingWitness, _compressed_filling,
                                   _boundary_before_value, _original_cell_birth)


class _CachedCycleLifter:
    """Maintain an increasing-filtration basis per degree; cache chosen cycles.

    Out-of-order queries rewind only the affected degree's basis, never the
    expensive global compression or persistence reduction.
    """

    def __init__(self, compressed, reduction, nodes, *, max_basis_columns: int,
                 max_basis_entries: int, max_chain_terms: int, max_work: int):
        self.compressed = compressed
        self.reduction = reduction
        self.nodes = nodes
        self.max_basis_columns = max_basis_columns
        self.max_basis_entries = max_basis_entries
        self.max_chain_terms = max_chain_terms
        self.max_work = max_work
        self.cache: dict[int, LiftedCycleRepresentative] = {}
        self.states = {}
        self.added_pair_columns = 0
        self.rewinds = 0

    def _new_state(self, dimension):
        self.rewinds += int(dimension in self.states)
        solver = _BoundarySolver(self.compressed.lift_trace.grid_shape,
                                 max_basis_columns=self.max_basis_columns,
                                 max_basis_entries=self.max_basis_entries)
        state = [None, 0, solver]  # current filtration level, trace position, solver
        self.states[dimension] = state
        return state

    def get_many(self, node_ids):
        ids = list(dict.fromkeys(int(i) for i in node_ids))
        missing = [i for i in ids if i not in self.cache]
        missing.sort(key=lambda i: (self.nodes[i].birth_value, self.nodes[i].dimension, i))
        trace = self.compressed.lift_trace
        pairs = trace.pairs if trace.pairs is not None else ()
        for node_id in missing:
            node = self.nodes[node_id]
            q, t = node.dimension, node.birth_value
            state = self.states.get(q)
            if state is None or (state[0] is not None and t < state[0]):
                state = self._new_state(q)
            while state[1] < len(pairs) and int(pairs[state[1]]['value']) <= t:
                pair = pairs[state[1]]
                if int(pair['dimension']) == q:
                    state[2].add(int(pair['death']))
                    self.added_pair_columns += 1
                state[1] += 1
            state[0] = t
            compressed_cycle = _compressed_v_column(self.reduction, node.birth_cell,
                                                      max_work=self.max_work)
            if any(int(self.compressed.dimensions[c]) != q for c in compressed_cycle):
                raise RuntimeError('Inconsistent dimension of compressed cycle')
            seed = self.compressed.original_grid_indices[compressed_cycle]
            lifted = state[2].lift(seed, max_chain_terms=self.max_chain_terms)
            coords = np.column_stack(np.unravel_index(lifted, trace.grid_shape))
            self.cache[node_id] = LiftedCycleRepresentative(
                q, node.birth_cell, node.death_cell, node.birth_value, node.death_value,
                np.asarray(compressed_cycle, dtype=np.int64), lifted, coords)
        return {i: self.cache[i] for i in ids}


class PPGAnalysis:
    """Compute once; investigate multiple exact death-event certificates.

    Use as a context manager. The cached cancellation trace is disk-backed and
    is removed by close(). `solver='component'` selectively explores only
    incidence components reachable from the boundary discrepancy, whereas
    `solver='pair-trace'` builds the entire eligible canceled-death basis.

    With retain_inclusion_trace=True, solver='replay' evaluates the exact
    inclusion map on selected cycles/fillings using optional projected
    cancellation tails, without assembling a global boundary basis.

    This API is experimental: worst-case selective dependency closures can
    still be global. It does not retain a dense explicit i, p, h.
    """

    def __init__(self, image, *, max_original_cells: int = 100_000_000,
                 temp_dir=None, reduction_storage='packed', accelerator='python',
                 max_basis_columns=200_000, max_basis_entries=5_000_000,
                 max_chain_terms=500_000, max_work=5_000_000,
                 max_relation_cache_entries=100_000,
                 max_cached_witnesses: int = 4,
                 retain_inclusion_trace: bool = False,
                 max_replay_columns: int = 100_000,
                 max_replay_visited: int = 5_000_000):
        image = np.asarray(image)
        if image.ndim not in (2, 3) or image.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
            raise ValueError('Expected a 2D/3D uint8 or uint16 scalar field')
        if any(v <= 0 for v in (max_original_cells, max_basis_columns, max_basis_entries,
                               max_chain_terms, max_work)):
            raise ValueError('All resource limits must be positive')
        self.image = image
        self.grid_shape = tuple(2 * int(s) + 1 for s in image.shape)
        self.compressed = None
        self.reduction = None
        self.closed = False
        self._lifter = None
        self._replay = None
        self._replay_cycle_cache = {}
        self.last_replay_stats = None
        self.retain_inclusion_trace = retain_inclusion_trace
        self.max_replay_columns = max_replay_columns
        self.max_replay_visited = max_replay_visited
        if max_replay_columns <= 0 or max_replay_visited <= 0:
            raise ValueError('Selective replay resource limits must be positive')
        if max_cached_witnesses < 0:
            raise ValueError('max_cached_witnesses must be nonnegative')
        self.max_cached_witnesses = max_cached_witnesses
        self._witness_cache = OrderedDict()
        self._event_targets = {}
        self.event_solves = 0
        self.max_relation_cache_entries = max_relation_cache_entries
        self.last_component_stats: ComponentSolveStats | None = None
        try:
            self.compressed = compress_discrete_tensor_streaming(
                image, max_cells=max_original_cells, temp_dir=temp_dir,
                retain_chain_lifts=True, retain_cell_provenance=True,
                retain_inclusion_trace=retain_inclusion_trace,
                accelerator=accelerator)
            self.reduction = reduce_boundary(self.compressed, storage=reduction_storage,
                                             log_mode='full', accelerator=accelerator,
                                             temp_dir=temp_dir)
            self.nodes = build_nodes(self.compressed.values, self.compressed.dimensions,
                                     self.reduction, storage='compact')
            self._lifter = _CachedCycleLifter(
                self.compressed, self.reduction, self.nodes,
                max_basis_columns=max_basis_columns, max_basis_entries=max_basis_entries,
                max_chain_terms=max_chain_terms, max_work=max_work)
            self.max_basis_columns = max_basis_columns
            self.max_basis_entries = max_basis_entries
            self.max_chain_terms = max_chain_terms
            self.max_work = max_work
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self.closed:
            self.closed = True
            if self.reduction is not None:
                self.reduction.close()
            if self.compressed is not None and self.compressed.lift_trace is not None:
                self.compressed.lift_trace.close()
            self._witness_cache.clear()
            self._replay_cycle_cache.clear()
            if self._replay is not None:
                self._replay.cache.clear()
                if self._replay.owner is not None:
                    self._replay.owner.clear()
                self._replay = None
            if self._lifter is not None:
                self._lifter.cache.clear()
                self._lifter.states.clear()

    def __enter__(self):
        if self.closed:
            raise RuntimeError('Analysis has already been closed')
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def cached_cycle_count(self):
        return len(self._lifter.cache)

    @property
    def cached_event_count(self):
        return len(self._event_targets)

    @property
    def pairs_replayed_for_cycles(self):
        return self._lifter.added_pair_columns

    def targets(self, source_node: int) -> tuple[int, ...]:
        if self.closed:
            raise RuntimeError('Analysis has been closed')
        if not 0 <= source_node < len(self.nodes):
            raise IndexError('Invalid persistence node')
        node = self.nodes[source_node]
        if node.essential:
            raise ValueError('Source must have finite death')
        key = (node.death_value, node.dimension)
        if key not in self._event_targets:
            grouped = defaultdict(set)
            for edge in iter_edges(self.compressed.values, self.reduction, self.nodes,
                                   event=key,
                                   max_relation_cache_entries=self.max_relation_cache_entries):
                grouped[edge.from_node].add(edge.to_node)
            self._event_targets[key] = grouped
            self.event_solves += 1
        return tuple(sorted(self._event_targets[key].get(source_node, ())))

    def replay_dependency_closure(self, source_node: int):
        """Measure the influence closure for a selected death-event filling.

        Includes only the query-local reverse inclusion replay of the
        compressed filling; excludes cycle reconstruction, preprocessing,
        factorization, and trace construction. The returned counts are
        observed cancellation dependencies, not a graph-wide bound.
        """
        if self.closed:
            raise RuntimeError('Analysis has been closed')
        if not self.retain_inclusion_trace:
            raise ValueError('Construct PPGAnalysis with retain_inclusion_trace=True')
        witness = self.witness(source_node, solver='replay')
        ids = (source_node, *witness.target_nodes)
        compressed_relation: set[int] = set()
        for node_id in ids:
            compressed_relation.symmetric_difference_update(
                map(int, self._replay_cycle_cache[node_id].compressed_cells))
        compressed_filling = _compressed_filling(
            self.compressed, self.reduction, compressed_relation,
            witness.event_value, max_work=self.max_work)
        lifting = self._replay.lift(
            self.compressed.original_grid_indices[compressed_filling],
            max_output_cells=self.max_chain_terms, collect_dependency_stats=True)
        if not np.array_equal(lifting, witness.filling_grid_indices):
            raise RuntimeError('Replay dependency audit disagrees with saved certificate')
        return self._replay.last_closure_stats

    def witness(self, source_node: int, *, solver='component',
                max_component_faces: int = 2_000_000) -> TransitionFillingWitness:
        if self.closed:
            raise RuntimeError('Analysis has been closed')
        if solver not in ('component', 'pair-trace', 'replay'):
            raise ValueError("solver must be 'component', 'pair-trace' or 'replay'")
        if solver == 'replay' and not self.retain_inclusion_trace:
            raise ValueError('Construct PPGAnalysis with retain_inclusion_trace=True')
        if not 0 <= source_node < len(self.nodes):
            raise IndexError('Invalid persistence node')
        cache_key = (source_node, solver)
        if cache_key in self._witness_cache:
            self._witness_cache.move_to_end(cache_key)
            return self._witness_cache[cache_key]
        node = self.nodes[source_node]
        if node.essential or node.dimension >= self.image.ndim:
            raise ValueError('Source must have finite death and a filling dimension')
        t = node.death_value
        targets = self.targets(source_node)
        ids = (source_node, *targets)
        if solver == 'replay':
            if self._replay is None:
                self._replay = SelectiveInclusionReplay(
                    self.compressed.lift_trace,
                    max_cached_columns=self.max_replay_columns,
                    max_visited_records=self.max_replay_visited,
                    max_influence_entries=self.max_basis_entries)
            reps = {}
            for node_id in ids:
                if node_id not in self._replay_cycle_cache:
                    selected = self.nodes[node_id]
                    compressed_cycle = _compressed_v_column(
                        self.reduction, selected.birth_cell, max_work=self.max_work)
                    lifted = self._replay.lift(
                        self.compressed.original_grid_indices[compressed_cycle],
                        max_output_cells=self.max_chain_terms)
                    if len(boundary_old_chain(lifted, self.grid_shape)):
                        raise RuntimeError('Reverse-replayed representative is not a cycle')
                    if any(_original_cell_birth(self.image, k, self.grid_shape)
                           > selected.birth_value for k in lifted):
                        raise RuntimeError('Reverse-replayed cycle exceeds birth filtration')
                    coords = np.column_stack(np.unravel_index(lifted, self.grid_shape))
                    self._replay_cycle_cache[node_id] = LiftedCycleRepresentative(
                        selected.dimension, selected.birth_cell, selected.death_cell,
                        selected.birth_value, selected.death_value,
                        np.asarray(compressed_cycle, dtype=np.int64), lifted, coords)
                reps[node_id] = self._replay_cycle_cache[node_id]
        else:
            reps = self._lifter.get_many(ids)
        original_relation: set[int] = set()
        compressed_relation: set[int] = set()
        for rep in reps.values():
            original_relation.symmetric_difference_update(map(int, rep.original_grid_indices))
            compressed_relation.symmetric_difference_update(map(int, rep.compressed_cells))
        relation = np.asarray(sorted(original_relation), dtype=np.int64)
        if len(boundary_old_chain(relation, self.grid_shape)):
            raise RuntimeError('Transition relation is not an original-grid cycle')
        compressed_filling = _compressed_filling(
            self.compressed, self.reduction, compressed_relation, t, max_work=self.max_work)
        if _boundary_before_value(self.compressed, self.reduction, compressed_relation, t):
            raise RuntimeError('Transition already bounded strictly before the death event')
        seed = set(map(int, self.compressed.original_grid_indices[compressed_filling]))
        residual = original_relation ^ set(map(int, boundary_old_chain(seed, self.grid_shape)))
        if solver == 'replay':
            self.last_component_stats = None
            filling = self._replay.lift(
                self.compressed.original_grid_indices[compressed_filling],
                max_output_cells=self.max_chain_terms)
            self.last_replay_stats = self._replay.last_stats
            correction = np.asarray(sorted(seed ^ set(map(int, filling))), dtype=np.int64)
        elif solver == 'component':
            self.last_replay_stats = None
            correction, self.last_component_stats = solve_component_correction(
                self.compressed.lift_trace, residual, dimension=node.dimension, level=t,
                max_basis_columns=self.max_basis_columns,
                max_basis_entries=self.max_basis_entries,
                max_chain_terms=self.max_chain_terms,
                max_component_faces=max_component_faces)
        elif solver == 'pair-trace':
            self.last_replay_stats = None
            self.last_component_stats = None
            pair_basis = _BoundarySolver(self.grid_shape,
                                         max_basis_columns=self.max_basis_columns,
                                         max_basis_entries=self.max_basis_entries)
            for pair in self.compressed.lift_trace.iter_pairs():
                if int(pair['dimension']) == node.dimension + 1 and int(pair['value']) <= t:
                    pair_basis.add(int(pair['death']))
            correction = pair_basis.solve_boundary(sorted(residual),
                                                    max_chain_terms=self.max_chain_terms)
        filling = np.asarray(sorted(seed ^ set(map(int, correction))), dtype=np.int64)
        if not np.array_equal(boundary_old_chain(filling, self.grid_shape), relation):
            raise RuntimeError('Incorrect original cubical boundary of witness')
        if not len(filling) or not any(_original_cell_birth(self.image, c, self.grid_shape) == t
                                      for c in filling):
            raise RuntimeError('Witness lacks an event-level filling cell')
        for c in filling:
            if _original_cell_birth(self.image, c, self.grid_shape) > t:
                raise RuntimeError('Witness exceeds event filtration level')
            if sum(v & 1 for v in np.unravel_index(int(c), self.grid_shape)) != node.dimension + 1:
                raise RuntimeError('Witness has incorrect cell dimension')
        coords = np.column_stack(np.unravel_index(filling, self.grid_shape))
        result = TransitionFillingWitness(
            int(source_node), targets, node.dimension, int(t), tuple(self.image.shape),
            reps[source_node], {k: reps[k] for k in targets}, relation, filling, coords,
            True, compressed_filling, correction, solver)
        if self.max_cached_witnesses:
            self._witness_cache[cache_key] = result
            while len(self._witness_cache) > self.max_cached_witnesses:
                self._witness_cache.popitem(last=False)
        return result
