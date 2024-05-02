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
    dt = 1e-2
    assert dt <= 1e-2
    fire = hoomd.md.minimize.FIRE(dt, 1e-7, 1.0, 1e-7)

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
    compare = np.array([0.9, 0, 0])
    if snap.communicator.rank == 0:
        pos = snap.particles.position[0]
        assert pos[0] > compare[0] and pos[0] < compare[0] + 0.01
        assert pos[1] == compare[1]
        assert pos[2] == compare[2]


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
    fire = hoomd.md.minimize.FIRE(1e-1, 1e-7, 1.0, 1e-7)

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

# make a hoomd force field that is a double well in 3D using CustomForce
class DoubleWell(hoomd.md.force.Custom):

    def __init__(self, x1=0.0, x2=-1.0, x3=0.0, x4=1.0, y2=1.0, z2=1.0):
        super().__init__()
        assert x2 < 0
        assert x4 > 0
        assert y2 > 0
        assert z2 > 0
        self._x1 = x1
        self._x2 = x2
        self._x3 = x3
        self._x4 = x4
        self._y2 = y2
        self._z2 = z2

    def _potential(self, pos, energy):
        # quadratic potential in y and z
        # double well in x, using quartic potential
        energy += self._x4*pos[:,0]**4 + self._x3*pos[:,0]**3 + self._x2*pos[:,0]**2 + self._x1*pos[:,0]
        energy += self._y2*pos[:,1]**2
        energy += self._z2*pos[:,2]**2
    
    def _force(self, pos, force):
        # force is the derivative of the potential
        force[:,0] = -4*self._x4*pos[:,0]**3 - 3*self._x3*pos[:,0]**2 - 2*self._x2*pos[:,0] - self._x1
        force[:,1] = -2*self._y2*pos[:,1]
        force[:,2] = -2*self._z2*pos[:,2]

    def set_forces(self, timestep):
        state: hoomd.State = self._simulation.state
        with self.cpu_local_force_arrays as arrays:
            with state.cpu_local_snapshot as snapshot:
                pos: hoomd.data.HOOMDArray = snapshot.particles.position
                self._force(pos._coerce_to_ndarray(), arrays.force._coerce_to_ndarray())
                self._potential(pos._coerce_to_ndarray(), arrays.potential_energy._coerce_to_ndarray())
                arrays.virial[:] = np.arange(6)[None, :]


def test_mimse_double_well(simulation_factory, one_particle_snapshot_factory):

    snap = one_particle_snapshot_factory()

    if snap.communicator.rank == 0:
        # minimium of the double well
        snap.particles.position[0] = [-0.5, 0, 0]
        snap.particles.velocity[0] = [0, 0, 0]
    sim: hoomd.Simulation = simulation_factory(snap)

    # Setup FIRE sim with Mimse force
    fire = hoomd.md.minimize.FIRE(1e-1, 1e-7, 1.0, 1e-7)

    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    mimse_force = mimse.Mimse(1.0, 1.0)
    double_well_force = DoubleWell(x1=-0.5)
    
    fire.forces.append(mimse_force)
    fire.forces.append(double_well_force)
    fire.methods.append(nve)
    sim.operations.integrator = fire

    # while not fire.converged:        
    sim.run(10000)


    bias_pos = np.array([[-0.5, 0, 0]])
    mimse_force.push_back(bias_pos)
    mimse_force.random_kick(0.01)

    fire.reset()
    while not fire.converged:        
        sim.run(10000)

    snap = sim.state.get_snapshot()
    compare = np.array([0.809017, 0, 0])
    if snap.communicator.rank == 0:
        np.testing.assert_array_almost_equal(snap.particles.position[0],
                                             compare,
                                             decimal=6)

class TiltedSin(hoomd.md.force.Custom):
    pass