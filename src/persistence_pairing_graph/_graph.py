"""Persistence-pairing graph construction from a canonical sparse reduction."""
from dataclasses import dataclass
from array import array
from collections.abc import Sequence
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from heapq import heapify, heappop, heappush

import numpy as np

from ._reduction import Reduction, _toggle


@dataclass(frozen=True, slots=True)
class PersistenceNode:
    dimension: int
    birth_cell: int
    death_cell: int | None
    birth_value: int | float
    death_value: int | float

    @property
    def persistence(self) -> float:
        return float(self.death_value) - float(self.birth_value)

    @property
    def essential(self) -> bool:
        return self.death_cell is None


@dataclass(frozen=True, slots=True)
class PersistenceEdge:
    dimension: int
    from_node: int
    to_node: int
    coefficient: int = 1


class CompactNodeTable(Sequence[PersistenceNode]):
    """Packed persistence endpoints, with dataclass records created on access.

    Stores two unsigned 32-bit cell indices per node instead of retaining
    millions of individually allocated Python records and scalar objects.
    Endpoint indices are canonical compressed filtration indices; the cell
    value/dimension arrays are shared with the compressed complex. 2**32-1
    denotes an essential death. Indexing or iterating remains API compatible.
    """
    ESSENTIAL = 0xFFFFFFFF

    def __init__(self, values: np.ndarray, dimensions: np.ndarray,
                 births: array, deaths: array):
        if len(births) != len(deaths):
            raise ValueError('Node birth/death arrays differ in length')
        if len(values) >= self.ESSENTIAL:
            raise ValueError('Compact node table supports fewer than 2**32-1 cells')
        self.values = values
        self.dimensions = dimensions
        self.births = births
        self.deaths = deaths

    def __len__(self) -> int:
        return len(self.births)

    @property
    def nbytes(self) -> int:
        """Bytes of owned endpoint arrays (underlying cell data is shared)."""
        return 4 * (len(self.births) + len(self.deaths))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[k] for k in range(*index.indices(len(self)))]
        birth = int(self.births[index])
        death = int(self.deaths[index])
        return PersistenceNode(int(self.dimensions[birth]), birth,
                               None if death == self.ESSENTIAL else death,
                               _scalar(self.values[birth]),
                               float('inf') if death == self.ESSENTIAL else _scalar(self.values[death]))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __eq__(self, other):
        if not isinstance(other, Sequence):
            return NotImplemented
        if len(self) != len(other):
            return False
        if isinstance(other, CompactNodeTable):
            if (self.births == other.births and self.deaths == other.deaths
                    and np.array_equal(self.values, other.values)
                    and np.array_equal(self.dimensions, other.dimensions)):
                return True
            return all(a == b for a, b in zip(self, other))
        return all(a == b for a, b in zip(self, other))

    def __repr__(self) -> str:
        return f'CompactNodeTable(n={len(self)}, endpoint_bytes={self.nbytes})'


def _scalar(value: np.generic) -> int | float:
    return value.item()


def build_nodes(values: np.ndarray, dims: np.ndarray, reduction: Reduction, *,
                storage: str = 'python') -> list[PersistenceNode] | CompactNodeTable:
    if storage not in ('python', 'compact'):
        raise ValueError("storage must be 'compact' or 'python'")
    if storage == 'compact':
        if len(values) >= CompactNodeTable.ESSENTIAL:
            raise ValueError('Too many generators for compact node storage')
        # Finite intervals are ordered by death column, then essential
        # intervals by birth index, matching the Rust and original Python API.
        death_of_birth = np.full(len(values), -1, dtype=np.int32)
        births = array('I')
        deaths = array('I')
        for j, pivot in enumerate(reduction.low):
            birth = int(pivot)
            if birth >= 0:
                death_of_birth[birth] = j
                if values[j] > values[birth]:
                    births.append(birth)
                    deaths.append(j)
        for birth in range(len(values)):
            if reduction.low[birth] < 0 and death_of_birth[birth] < 0:
                births.append(birth)
                deaths.append(CompactNodeTable.ESSENTIAL)
        return CompactNodeTable(values, dims, births, deaths)
    death_of_birth = np.full(len(values), -1, dtype=np.int64)
    nodes = []
    for death, birth in enumerate(reduction.low):
        if birth >= 0:
            death_of_birth[birth] = death
            if values[death] > values[birth]:
                nodes.append(PersistenceNode(int(dims[birth]), int(birth), death,
                                             _scalar(values[birth]), _scalar(values[death])))
    for birth in range(len(values)):
        if reduction.low[birth] < 0 and death_of_birth[birth] < 0:
            nodes.append(PersistenceNode(int(dims[birth]), birth, None,
                                         _scalar(values[birth]), float('inf')))
    return nodes




def _solve_triangular_bitsets(rows: list[set[int]], m: int) -> list[int] | None:
    """Return compact right-block rows after exact GF(2) forward substitution.

    Python arbitrary-precision integers execute wide XORs in native code.
    The dying-coordinate block must be unit lower triangular; return None
    to let the generic event solver handle an unexpected nontriangular block.
    """
    left_mask = (1 << m) - 1
    packed = []
    for i, row in enumerate(rows):
        bits = 0
        for j in row:
            bits |= 1 << j
        if not (bits & (1 << i)) or ((bits & left_mask) >> (i + 1)):
            return None
        packed.append(bits)
    for i in range(m):
        lower = packed[i] & ((1 << i) - 1)
        while lower:
            first = lower & -lower
            lower ^= first
            packed[i] ^= packed[first.bit_length() - 1]
        if (packed[i] & left_mask) != (1 << i):
            raise RuntimeError('Bitset event solver failed to resolve dying block')
    return [bits >> m for bits in packed]

def iter_edges(values: np.ndarray, reduction: Reduction,
               nodes: Sequence[PersistenceNode], *,
               max_relation_cache_entries: int = 100_000,
               skip_empty_survivor_events: bool = True):
    """Compute directed transition coefficients with simultaneous-death events.

    For each death event, relations are projected into the active persistence
    basis immediately before the event, then [C_dying | C_surviving] is solved
    to [I | W] over F2. Nonzero W entries are the graph edges.
    """
    if max_relation_cache_entries < 0:
        raise ValueError('max_relation_cache_entries must be nonnegative')
    n = len(values)
    death_of_birth = np.full(n, -1, dtype=np.int32)
    for death, birth in enumerate(reduction.low):
        if birth >= 0:
            death_of_birth[birth] = death
    if isinstance(nodes, CompactNodeTable):
        # An O(number of generators) packed lookup avoids one dictionary entry
        # per persistence node (tens of bytes per entry in CPython).
        birth_to_node = np.full(n, -1, dtype=np.int32)
        node_births = np.frombuffer(nodes.births, dtype=np.uint32)
        node_deaths = np.frombuffer(nodes.deaths, dtype=np.uint32)
        birth_to_node[node_births] = np.arange(len(nodes), dtype=np.int32)
        node_dims = nodes.dimensions[node_births]
        finite_count = int(np.count_nonzero(node_deaths != CompactNodeTable.ESSENTIAL))
        birth_times = values[node_births]
        finite_death_times = values[node_deaths[:finite_count]]
        finite = np.lexsort((node_births[:finite_count], node_dims[:finite_count],
                             finite_death_times))
        def birth_at(k): return int(node_births[k])
        def dim_at(k): return int(node_dims[k])
        def birth_value_at(k): return _scalar(birth_times[k])
        def death_value_at(k):
            return (float('inf') if k >= finite_count
                    else _scalar(finite_death_times[k]))
        def lookup(birth):
            k = int(birth_to_node[birth]); return None if k < 0 else k
    else:
        birth_to_node = {node.birth_cell: k for k, node in enumerate(nodes)}
        finite = sorted((k for k, node in enumerate(nodes) if not node.essential),
                        key=lambda k: (nodes[k].death_value, nodes[k].dimension, nodes[k].birth_cell))
        def birth_at(k): return nodes[k].birth_cell
        def dim_at(k): return nodes[k].dimension
        def birth_value_at(k): return nodes[k].birth_value
        def death_value_at(k): return nodes[k].death_value
        def lookup(birth): return birth_to_node.get(birth)

    # Exact active interval counting: a class is available as a target only
    # if birth < t < death. Unlike checking the latest death, this also
    # recognizes gaps before the next birth in the same dimension.
    births_by_dim: dict[int, list | np.ndarray] = {}
    deaths_by_dim: dict[int, list | np.ndarray] = {}
    if skip_empty_survivor_events:
        if isinstance(nodes, CompactNodeTable):
            # Keep the birth/death event indexes in the original scalar dtype;
            # no per-node Python tuples, floats or lists are created.
            for d in np.unique(node_dims):
                selected = node_dims == d
                births_by_dim[int(d)] = np.sort(values[node_births[selected]])
                finite_mask = selected & (node_deaths != CompactNodeTable.ESSENTIAL)
                deaths_by_dim[int(d)] = np.sort(values[node_deaths[finite_mask]])
        else:
            for k in range(len(nodes)):
                d = dim_at(k)
                births_by_dim.setdefault(d, []).append(birth_value_at(k))
                if death_value_at(k) != float('inf'):
                    deaths_by_dim.setdefault(d, []).append(death_value_at(k))
            for seq in births_by_dim.values(): seq.sort()
            for seq in deaths_by_dim.values(): seq.sort()
    cache: OrderedDict[int, list[int]] = OrderedDict()
    cache_words = 0

    def relation(birth: int) -> list[int]:
        nonlocal cache_words
        if birth in cache:
            cache.move_to_end(birth)
            return cache[birth]
        death = int(death_of_birth[birth])
        if death < 0:
            raise RuntimeError(f"Cell {birth} has no death relation")
        reduced = reduction.reduced_columns[death]
        # V^{-1} R_death: the original R support determines which operation
        # lists act; newly introduced coordinates are not visited recursively.
        coords = set(reduced)
        for target in reduced:
            for source in reduction.operations.get(target, ()):
                _toggle(coords, source)
        result = sorted(coords)
        if not result or result[-1] != birth:
            raise RuntimeError("Death relation lacks its expected leading birth")
        if any(reduction.low[idx] >= 0 for idx in result):
            raise RuntimeError("Death relation contains a negative persistence generator")
        if max_relation_cache_entries and len(result) <= max_relation_cache_entries:
            old = cache.pop(birth, None)
            if old is not None:
                cache_words -= len(old)
            while cache and cache_words + len(result) > max_relation_cache_entries:
                _, evicted = cache.popitem(last=False)
                cache_words -= len(evicted)
            cache[birth] = result
            cache_words += len(result)
        return result

    def project_active(raw: list[int], t: int | float) -> list[int]:
        pending = set(raw)
        heap = [-x for x in raw]
        heapify(heap)
        active = []
        while heap:
            index = -heappop(heap)
            if index not in pending:
                continue
            pending.remove(index)
            death = int(death_of_birth[index])
            if death >= 0 and values[death] < t:
                for lower in relation(index)[:-1]:
                    if lower in pending:
                        pending.remove(lower)
                    else:
                        pending.add(lower)
                        heappush(heap, -lower)
            else:
                active.append(index)
        active.reverse()
        return active

    start = 0
    while start < len(finite):
        t, dim = death_value_at(finite[start]), dim_at(finite[start])
        end = start + 1
        while end < len(finite) and (death_value_at(finite[end]) == t
                                     and dim_at(finite[end]) == dim):
            end += 1
        if skip_empty_survivor_events:
            if isinstance(nodes, CompactNodeTable):
                active_after = (int(np.searchsorted(births_by_dim[dim], t, side='left'))
                                - int(np.searchsorted(deaths_by_dim[dim], t, side='right')))
            else:
                active_after = (bisect_left(births_by_dim[dim], t)
                                - bisect_right(deaths_by_dim[dim], t))
        else:
            active_after = -1
        if active_after == 0:
            start = end
            continue
        dying = sorted(finite[start:end], key=birth_at)
        m = len(dying)
        dying_positions = {birth_at(k): i for i, k in enumerate(dying)}
        projected_rows = []
        touched: set[int] = set()
        for node_index in dying:
            rows = project_active(relation(birth_at(node_index)), t)
            projected_rows.append(rows)
            for birth in rows:
                if birth not in dying_positions:
                    survivor_index = lookup(birth)
                    if survivor_index is None:
                        raise RuntimeError(f"Active birth {birth} is not a graph node")
                    if not (dim_at(survivor_index) == dim and
                            birth_value_at(survivor_index) < t < death_value_at(survivor_index)):
                        raise RuntimeError("Projected relation contains a non-surviving interval")
                    touched.add(birth)
        survivors = sorted(touched)
        survivor_positions = {birth: m + i for i, birth in enumerate(survivors)}
        rows: list[set[int]] = [
            {dying_positions[x] if x in dying_positions else survivor_positions[x] for x in row}
            for row in projected_rows
        ]
        # Compact event solver: Python int XOR operates on many F2 entries in
        # native machine words. Keep a cap for unusually enormous events.
        estimated_bitset_bytes = (m * (m + len(survivors)) + 7) // 8
        nonzeros = sum(map(len, rows))
        density = nonzeros / max(1, m * (m + len(survivors)))
        if (m >= 128 and estimated_bitset_bytes <= 32 * 1024 * 1024
                and (nonzeros >= 16_384 or density >= 0.02)):
            right_rows = _solve_triangular_bitsets(rows, m)
            if right_rows is not None:
                for i, right in enumerate(right_rows):
                    while right:
                        bit = right & -right
                        right ^= bit
                        j = bit.bit_length() - 1
                        yield PersistenceEdge(dim, dying[i], lookup(survivors[j]))
                start = end
                continue
        # The dying block is unit lower triangular when births are ordered by
        # increasing filtration index; use forward substitution when possible.
        triangular = all(i in row and not any(i < x < m for x in row)
                         for i, row in enumerate(rows))
        if triangular:
            for i in range(m):
                for lower in sorted(x for x in rows[i] if x < i):
                    rows[i].symmetric_difference_update(rows[lower])
        else:
            # Defensive generic Gauss--Jordan solver for any invertible block.
            for column in range(m):
                pivot = next((r for r in range(column, m) if column in rows[r]), None)
                if pivot is None:
                    raise RuntimeError("Singular simultaneous-death transition block")
                rows[column], rows[pivot] = rows[pivot], rows[column]
                for r in range(m):
                    if r != column and column in rows[r]:
                        rows[r].symmetric_difference_update(rows[column])
        for i, row in enumerate(rows):
            if {x for x in row if x < m} != {i}:
                raise RuntimeError("Could not resolve simultaneous-death transition")
            for j in sorted(x for x in row if x >= m):
                yield PersistenceEdge(dim, dying[i], lookup(survivors[j - m]))
        start = end


def build_edges(values: np.ndarray, reduction: Reduction,
                nodes: Sequence[PersistenceNode], *,
                max_relation_cache_entries: int = 100_000) -> list[PersistenceEdge]:
    """Materialize directed edges; use iter_edges for constant-output-buffer export."""
    return list(iter_edges(values, reduction, nodes,
                           max_relation_cache_entries=max_relation_cache_entries))
