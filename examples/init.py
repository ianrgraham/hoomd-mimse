import argparse

from typing import Iterable, List, Optional, Sequence, Union, Tuple, Callable, Any
from inspect import signature

import numpy as np
from numpy.typing import ArrayLike
import hoomd
import gsd.hoomd

def init_rng(
    seed: Union[int, Sequence[int]]
) -> Union[np.random.Generator, Sequence[np.random.Generator]]:
    '''Simple helper function to spawn random number generators.'''
    return np.random.default_rng(seed)


def L_box_from_rho(N: int, rho: float, dim: int = 3):
    """Calculate regular box length for a given particle density"""
    assert dim in [2, 3], "Valid dims in hoomd are 2 and 3"
    return np.power(N / rho, 1 / dim)


def L_box_from_vol_frac(diams: ArrayLike, vol_frac: float, dim: int = 3):
    """Calculate regular box length for a given volume fraction"""
    assert dim in [2, 3], "Valid dims in hoomd are 2 and 3"
    part_vol = np.sum(np.square(diams / 2) * np.pi)
    return np.power(part_vol / vol_frac, 1 / dim)


def approx_euclidean_snapshot(
        N: int,
        L: float,
        rng: np.random.Generator,
        dim: int = 3,
        particle_types: Optional[List[str]] = None,
        ratios: Optional[List[int]] = None,
        diams: Optional[List[float]] = None) -> gsd.hoomd.Frame:
    '''Constucts hoomd simulation snapshot with regularly spaced particles on a
    euclidian lattice.

    Easy way to initialize simulations states for pair potentials like
    Lennard-Jones where large particle overlaps can leading to particles being
    ejected from the box. Simulation should be thoroughly equilibrated after
    such setup.

    Arguments
    ---------
        `N`: Number of particles.
        `L`: Side length of the simulation box.
        `rng`: `numpy` RNG to choose where to place species.
        `dim`: Physical dimension of the box (default=3).
        `particle_types`: List of particle labels (default=['A', 'B']).
        `ratios`: List of particle ratios (default=[50, 50]).
        `diams`: List of particle diameters for visualization.

    Returns
    -------
        `Frame`: A valid `hoomd` simulation state.
    '''

    if particle_types is None:
        particle_types = ['A', 'B']

    if ratios is None:
        ratios = [50, 50]

    # only valid dims in hoomd are 2 and 3
    assert dim in [2, 3], "Valid dims in hoomd are 2 and 3"
    assert L > 0, "Box length cannot be <= 0"
    assert N > 0, "Number of particles cannot be <= 0"
    len_types = len(particle_types)
    assert np.sum(ratios) == 100, "Ratios must sum to 100"
    assert len_types == len(
        ratios), "Lens of 'particle_types' and 'ratios' must match"
    if diams is not None:
        assert len_types == len(diams)
    
    if diams is None:
        diams = [1.0] * len_types

    n = int(np.ceil(np.power(N, 1 / dim)))
    x = np.linspace(-L / 2, L / 2, n, endpoint=False)
    X = [x for _ in range(dim)]
    if dim == 2:
        X.append(np.zeros(1))
    grid = np.meshgrid(*X)
    grid = [x.flatten() for x in grid]
    pos = np.stack(grid, axis=-1)

    if dim == 2:
        Lz = 0.0
    else:
        Lz = L
    # build snapshot and populate particles positions
    snapshot = gsd.hoomd.Frame()
    snapshot.particles.N = N
    snapshot.particles.position = pos[:N]
    snapshot.configuration.box = [L, L, Lz, 0.0, 0.0, 0.0]
    snapshot.particles.types = particle_types
    snapshot.particles.typeid = [0] * N
    snapshot.particles.diameter = [0] * N

    # assign particle labels with rng
    idx = 0
    limits = np.cumsum(ratios)
    j = 0
    for i in rng.permutation(np.arange(N)):
        while j / N * 100 >= limits[idx]:
            idx += 1
        snapshot.particles.typeid[i] = idx
        snapshot.particles.diameter[i] = diams[idx]

        j += 1

    return snapshot
