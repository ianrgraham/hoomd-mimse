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

    snap = gsd.hoomd.Frame()
    
    pos = np.loadtxt(this_dir / f"data/validate-pos-{iter}.dat") - L/2
    force = np.loadtxt(this_dir / f"data/validate-force-{iter}.dat")
    bias_pos = np.loadtxt(this_dir / f"data/validate-bias-{iter}.dat") - L/2

    snap.particles.N = N
    snap.particles.position = np.zeros((N, 3))
    snap.particles.velocity = np.zeros((N, 3))
    snap.particles.types = ['A']
    snap.particles.typeid = [0] * N
    snap.configuration.box = [L, L, L, 0, 0, 0]
    sim: hoomd.Simulation = simulation_factory(snap)

    # Setup FIRE sim with Mimse force
    dt = 1e-2
    assert dt <= 1e-2
    fire = hoomd.md.minimize.FIRE(dt, 1e-7, 1.0, 1e-7)

    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    mimse_force = mimse.Mimse(1.25, 20.0, subtract_mean=True)
    
    fire.forces.append(mimse_force)
    fire.methods.append(nve)
    sim.operations.integrator = fire
    sim.run(0)

    if sim.device.communicator.rank == 0:
        snap = sim.state.get_snapshot()
        snap.particles.position[:] = pos
        sim.state.set_snapshot(snap)
    mimse_force.push_back(bias_pos)

    sim.run(0)

    bias_force = mimse_force.forces

    mean_bias_force = np.mean(bias_force, axis=0)
    mean_force = np.mean(force, axis=0)
    np.testing.assert_allclose(mean_bias_force, np.zeros(3), atol=1e-20)
    np.testing.assert_allclose(mean_force, np.zeros(3), atol=1e-20)


    np.testing.assert_allclose(bias_force, force, rtol=1e-10, atol=1e-10)