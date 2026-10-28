"""Relative localization: old-chain correction is global; new cells are restricted."""
from itertools import combinations
from math import prod

import numpy as np
import pytest

from persistence_pairing_graph import (
    certify_local_filling, certify_relative_filling,
    certify_transition_relative_locality, transition_filling_witness,
)
from persistence_pairing_graph._lift import boundary_old_chain
from persistence_pairing_graph._transition_witness import _original_cell_birth


def _cells(image, dimension, before_or_at):
    grid = tuple(2*n+1 for n in image.shape)
    cells = []
    for c in range(prod(grid)):
        if sum(x & 1 for x in np.unravel_index(c, grid)) != dimension:
            continue
        if before_or_at(_original_cell_birth(image, c, grid)):
            cells.append(c)
    return cells


def _bruteforce(relation, old, new, grid):
    allowed = old + new
    assert len(allowed) <= 15
    target = set(map(int, relation))
    return any(set(map(int, boundary_old_chain(subset, grid))) == target
               for k in range(len(allowed)+1)
               for subset in combinations(allowed, k))


def test_old_correction_needed_to_fill_outer_ring():
    image = np.ones((3, 3), dtype=np.uint8)
    image[1, 1] = 2
    grid = (7, 7)
    old = _cells(image, 2, lambda t: t < 2)
    new = _cells(image, 2, lambda t: t == 2)
    assert len(old) == 8 and len(new) == 1
    relation = boundary_old_chain(old + new, grid)
    # The outer perimeter is an OLD cycle. The central new square alone
    # cannot fill it; old squares correct its boundary exactly.
    assert all(_original_cell_birth(image, int(c), grid) < 2 for c in relation)
    absolute = certify_local_filling(relation, new, image.shape, dimension=1)
    assert not absolute.feasible
    relative = certify_relative_filling(image, relation, new, dimension=1,
                                       event_value=2)
    assert relative.feasible
    assert len(relative.old_filling_cells) == 8
    assert len(relative.new_filling_cells) == 1
    filling = list(relative.old_filling_cells)+list(relative.new_filling_cells)
    assert np.array_equal(boundary_old_chain(filling, grid), relation)
    missing = certify_relative_filling(image, relation, [], dimension=1,
                                       event_value=2)
    assert not missing.feasible
    obstruction = set(missing.obstruction_cells)
    assert len(obstruction.intersection(relation)) % 2 == 1
    assert all(len(obstruction.intersection(boundary_old_chain([c],grid))) % 2 == 0
               for c in old)


def test_box_restricts_only_new_filling_cells():
    image = np.ones((3,3), dtype=np.uint8)
    image[1,1] = 2
    grid = (7,7)
    relation = boundary_old_chain(_cells(image, 2, lambda t: t <= 2), grid)
    class Witness:
        image_shape = (3,3)
        dimension = 1
        event_value = 2
        relation_support = relation
    excluded = certify_transition_relative_locality(
        image, Witness(), lower=(0,0), upper=(1,1))
    assert not excluded.feasible
    near_center = certify_transition_relative_locality(
        image, Witness(), lower=(3,3), upper=(3,3))
    assert near_center.feasible
    assert near_center.old_candidate_count == 8
    assert near_center.new_candidate_count == 1


@pytest.mark.parametrize('shape', [(2,2), (2,1,1), (1,2,1)])
def test_randomized_exact_against_bruteforce(shape):
    rng = np.random.default_rng(311+len(shape)+sum(shape))
    grid = tuple(2*n+1 for n in shape)
    q = 1
    for _ in range(40):
        image = rng.integers(0, 3, size=shape, dtype=np.uint8)
        t = 1
        old = _cells(image, q+1, lambda x: x < t)
        new = _cells(image, q+1, lambda x: x == t)
        selected = [x for x in new if rng.random() < 0.65]
        # Generate old-cycle targets via arbitrary old/new combinations;
        # retain only those whose boundaries lie entirely in A.
        support = [x for x in old+new if rng.random() < 0.5]
        relation = boundary_old_chain(support, grid)
        if any(_original_cell_birth(image,int(c),grid) >= t for c in relation):
            continue
        if len(old)+len(selected) > 15:
            continue
        cert = certify_relative_filling(
            image, relation, selected, dimension=q, event_value=t)
        expected = _bruteforce(relation, old, selected, grid)
        assert cert.feasible == expected
        if expected:
            actual = [*cert.old_filling_cells,*cert.new_filling_cells]
            assert np.array_equal(boundary_old_chain(actual,grid), relation)
        else:
            dual=set(cert.obstruction_cells)
            assert len(dual.intersection(relation)) % 2 == 1
            assert all(len(dual.intersection(boundary_old_chain([x],grid))) % 2 == 0
                       for x in old+selected)


def test_real_pair_of_pants_relative_certificate():
    image = np.asarray([
        [0,0,0,0,0,0,0,0,0,0,0],
        [0,30,30,30,30,30,30,30,30,30,0],
        [0,30,10,10,10,30,20,20,20,30,0],
        [0,30,10,90,10,30,20,80,20,30,0],
        [0,30,10,10,10,30,20,20,20,30,0],
        [0,30,30,30,30,30,30,30,30,30,0],
        [0,0,0,0,0,0,0,0,0,0,0],
    ],dtype=np.uint8)
    witness=transition_filling_witness(image,2)
    cert=certify_transition_relative_locality(
        image,witness,lower=(0,0),upper=(14,22))
    assert cert.feasible
    assert len(cert.new_filling_cells)>0
    assert np.array_equal(
        boundary_old_chain([*cert.old_filling_cells,*cert.new_filling_cells], (15,23)),
        witness.relation_support)


def test_invalid_filtration_membership_and_limits():
    image=np.asarray([[1,2],[1,1]],dtype=np.uint8)
    grid=(5,5)
    new=_cells(image,2,lambda x:x==2)
    old=_cells(image,2,lambda x:x<2)
    if new:
        with pytest.raises(ValueError,match='exactly'):
            certify_relative_filling(image,[],old[:1],dimension=1,event_value=2)
        with pytest.raises(ValueError,match='K_'):
            certify_relative_filling(
                image,boundary_old_chain(new,grid),new,dimension=1,event_value=2)
    with pytest.raises(MemoryError,match='max_grid_cells'):
        certify_relative_filling(image,[],new,dimension=1,event_value=2,
                                 max_grid_cells=2)
