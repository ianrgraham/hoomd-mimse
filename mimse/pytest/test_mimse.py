# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

# Import the plugin module.
from hoomd.mimse import mimse

# Import the hoomd Python package.
import hoomd
from hoomd import operation
from hoomd.conftest import *

import itertools
import pytest
import numpy as np


# @pytest.mark.parametrize("vel", [[0, 0, 0]])
def test_mimse(simulation_factory, one_particle_snapshot_factory):

    # `one_particle_snapshot_factory` and `simulation_factory` are pytest
    # fixtures defined in hoomd/conftest.py. These factories automatically
    # handle iterating tests over different CPU and GPU devices.
    snap = one_particle_snapshot_factory()

    if snap.communicator.rank == 0:
        snap.particles.position[0] = [0, 0, 0]
        snap.particles.velocity[0] = [0, 0, 0]
    sim: hoomd.Simulation = simulation_factory(snap)

    # Setup FIRE sim with Mimse force
    fire = hoomd.md.minimize.FIRE(1e-3, 1e-7, 1.0, 1e-5)

    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    force = mimse.Mimse(1.0, 1.0)
    
    fire.forces.append(force)
    fire.methods.append(nve)
    sim.operations.integrator = fire

    # Test that the initial snapshot is correct
    sim.run(0)
    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        np.testing.assert_array_almost_equal(snap.particles.position[0],
                                             np.array([0, 0, 0]),
                                             decimal=6)

    bias_pos = np.array([[-0.1, 0, 0]])
    force.push_back(bias_pos)

    while not fire.converged:        
        sim.run(10000)

    snap = sim.state.get_snapshot()
    compare = np.array([0.900713, 0, 0])
    if snap.communicator.rank == 0:
        np.testing.assert_array_almost_equal(snap.particles.position[0],
                                             compare,
                                             decimal=6)


def test_mimse_push_and_prune(simulation_factory, one_particle_snapshot_factory):

    # `one_particle_snapshot_factory` and `simulation_factory` are pytest
    # fixtures defined in hoomd/conftest.py. These factories automatically
    # handle iterating tests over different CPU and GPU devices.
    snap = one_particle_snapshot_factory()

    if snap.communicator.rank == 0:
        snap.particles.position[0] = [0, 0, 0]
        snap.particles.velocity[0] = [0, 0, 0]
    sim: hoomd.Simulation = simulation_factory(snap)

    # Setup FIRE sim with Mimse force
    fire = hoomd.md.minimize.FIRE(1e-3, 1e-7, 1.0, 1e-5)

    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    force = mimse.Mimse(1.0, 1.0)
    
    fire.forces.append(force)
    fire.methods.append(nve)
    sim.operations.integrator = fire

    # Test that the initial snapshot is correct
    sim.run(0)

    bias_pos = np.array([[-2, 0, 0]])
    force.push_back(bias_pos)
    if snap.communicator.rank == 0:
        bias = force.get_biases()
        assert len(bias) == 1
        assert bias[0][0, 0] == -2

    bias_pos = np.array([[0.5, 0, 0]])
    force.push_back(bias_pos)
    if snap.communicator.rank == 0:
        bias = force.get_biases()
        assert len(bias) == 2
        assert bias[1][0, 0] == 0.5

    bias_pos = np.array([[1, 1, 0]])
    force.push_back(bias_pos)
    bias_pos = np.array([[0, 0.25, 0]])
    force.push_back(bias_pos)

    force.prune_biases(1.0)
    if snap.communicator.rank == 0:
        bias = force.get_biases()
        assert len(bias) == 2
        np.testing.assert_array_almost_equal(bias[0][0], np.array([0.5, 0, 0]))
        np.testing.assert_array_almost_equal(bias[1][0], np.array([0, 0.25, 0]))
