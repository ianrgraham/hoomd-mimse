# Import the plugin module.
from hoomd.mimse import mimse

# Import the hoomd Python package.
import hoomd
from hoomd import operation
from hoomd.conftest import *

import itertools
import pytest
import numpy as np
import gsd.hoomd
import pathlib

iters = [1, 2]
Ns = [256, 64]
Ls = [5.975206328742890, 5.975206328742890]

testdata = list(zip(iters, Ns, Ls))

@pytest.mark.parametrize("iter, N, L", testdata)
def test_validate_existing(simulation_factory, iter, N, L):
    """
    Validate the our mimse code matches prior results."""

    this_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))

    snap = gsd.hoomd.open((this_dir / f"final_molecule_quench.gsd").as_posix())[-1]
    sim: hoomd.Simulation = simulation_factory(snap)

    # make sim verbose
    sim.device.notice_level = 4

    # Setup FIRE sim with Mimse force
    dt = 1e-2
    assert dt <= 1e-2
    fire = hoomd.md.minimize.FIRE(dt, 1e-7, 1.0, 1e-7)

    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    mimse_force = mimse.Mimse(1.25, 20.0, subtract_mean=True, mode="molecule")
    
    fire.forces.append(mimse_force)
    fire.methods.append(nve)
    sim.operations.integrator = fire
    sim.run(0)
