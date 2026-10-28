"""Exact local filling or dual obstruction, independently checked over GF(2)."""
from itertools import combinations
from math import prod

import numpy as np
import pytest

from persistence_pairing_graph import (
    certify_local_filling, certify_transition_filling_locality,
    eligible_filling_cells_in_box, transition_filling_witness,
)
from persistence_pairing_graph._lift import boundary_old_chain


def _cells(image_shape, dimension):
    shape = tuple(2*x+1 for x in image_shape)
    return [i for i in range(prod(shape))
            if sum(int(c) & 1 for c in np.unravel_index(i, shape)) == dimension]


def test_single_square_filling():
    shape = (1, 1)
    square = _cells(shape, 2)[0]
    boundary = boundary_old_chain([square], (3, 3))
    certificate = certify_local_filling(boundary, [square], shape, dimension=1)
    assert certificate.feasible
    assert list(certificate.filling_cells) == [square]
    assert len(certificate.obstruction_cells) == 0


def test_missing_square_has_a_dual_obstruction():
    shape = (2, 2)
    grid = (5, 5)
    square, other = _cells(shape, 2)[:2]
    boundary = boundary_old_chain([square], grid)
    cert = certify_local_filling(boundary, [other], shape, dimension=1)
    assert not cert.feasible
    obstruction = set(cert.obstruction_cells)
    assert len(obstruction.intersection(boundary)) % 2 == 1
    assert len(obstruction.intersection(boundary_old_chain([other], grid))) % 2 == 0


def test_brute_force_random_2d_and_3d():
    rng = np.random.default_rng(743)
    for shape, q, iterations in [
        ((1, 1), 1, 35),
        ((2, 2), 1, 100),
        ((1, 1, 1), 1, 80),
        ((1, 1, 1), 2, 80),
    ]:
        grid = tuple(2*x + 1 for x in shape)
        pool = _cells(shape, q+1)
        for _ in range(iterations):
            allowed = sorted(rng.choice(
                pool, size=int(rng.integers(0, min(5, len(pool))+1)),
                replace=False).tolist())
            support = [v for v in pool if rng.random() < 0.5]
            target = list(boundary_old_chain(support, grid))
            cert = certify_local_filling(target, allowed, shape, dimension=q)
            brute = any(
                list(boundary_old_chain(subset, grid)) == target
                for size in range(len(allowed)+1)
                for subset in combinations(allowed, size))
            assert cert.feasible == brute
            if cert.feasible:
                assert list(boundary_old_chain(cert.filling_cells, grid)) == target
            else:
                obstruction = set(cert.obstruction_cells)
                assert len(obstruction.intersection(target)) % 2 == 1
                assert all(
                    len(obstruction.intersection(boundary_old_chain([a], grid))) % 2 == 0
                    for a in allowed)


def test_box_filtering_for_toy_relation():
    image = np.asarray([[0, 3], [3, 3]], dtype=np.uint8)
    square = _cells(image.shape, 2)[0]
    boundary = boundary_old_chain([square], (5, 5))

    class Witness:
        image_shape = (2, 2)
        dimension = 1
        event_value = 3
        relation_support = boundary

    allowed = eligible_filling_cells_in_box(
        image, dimension=1, event_value=3, lower=(0, 0), upper=(4, 4))
    assert square in allowed
    whole = certify_transition_filling_locality(
        image, Witness(), lower=(0, 0), upper=(4, 4))
    assert whole.feasible
    restricted = certify_transition_filling_locality(
        image, Witness(), lower=(0, 0), upper=(0, 0))
    assert not restricted.feasible


def test_integration_with_real_complete_ppg_death_relation():
    image = np.asarray([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 0],
        [0, 30, 10, 10, 10, 30, 20, 20, 20, 30, 0],
        [0, 30, 10, 90, 10, 30, 20, 80, 20, 30, 0],
        [0, 30, 10, 10, 10, 30, 20, 20, 20, 30, 0],
        [0, 30, 30, 30, 30, 30, 30, 30, 30, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)
    witness = transition_filling_witness(image, 2)
    # The existing verified filling must be feasible under its own support.
    selected = certify_local_filling(
        witness.relation_support, witness.filling_grid_indices, image.shape,
        dimension=witness.dimension)
    assert selected.feasible
    assert np.array_equal(
        boundary_old_chain(selected.filling_cells, (15, 23)),
        witness.relation_support)
    # The whole grid at the death threshold must also permit a filling.
    full = certify_transition_filling_locality(
        image, witness, lower=(0, 0), upper=(14, 22))
    assert full.feasible


def test_bad_input_and_bounded_work():
    with pytest.raises(ValueError):
        certify_local_filling([1], [1], (1, 1), dimension=1)
    with pytest.raises(MemoryError):
        certify_local_filling([], _cells((2, 2), 2), (2, 2),
                              dimension=1, max_allowed_cells=1)
    with pytest.raises(ValueError):
        eligible_filling_cells_in_box(
            np.zeros((2, 2), dtype=np.uint8),
            dimension=1, event_value=0, lower=(-1, 0), upper=(4, 4))
    with pytest.raises(MemoryError):
        eligible_filling_cells_in_box(
            np.zeros((2, 2), dtype=np.uint8),
            dimension=1, event_value=0, lower=(0, 0), upper=(4, 4),
            max_examined_cells=4)
