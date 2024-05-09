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
import gsd.hoomd
import numpy as np

class TiltedSinVec(hoomd.md.force.Custom):
    # ux = sin(x) - 0.8*x
    # local minima: -0.643501, 5.63968, 11.9228653072 

    def __init__(self, v1, v2, v3):
        super().__init__()
        self._v1 = v1
        self._v2 = v2
        self._v3 = v3
    
    def _u(self, v1, v2, v3, x):
        x1 = np.sum(x*v1, axis=-1)
        x2 = np.sum(x*v2, axis=-1)
        x3 = np.sum(x*v3, axis=-1)
        return 0 # (np.sin(x1) - 0.8*x1) + x2*x2 + x3*x3
    
    def _f(self, v1, v2, v3, x):
        x1 = np.sum(x*v1, axis=-1)
        x2 = np.sum(x*v2, axis=-1)
        x3 = np.sum(x*v3, axis=-1)
        tmp = v1.T*(-np.cos(x1) + 0.8) - 2*v2.T*x2 - 2*v3.T*x3
        return tmp.T

    def _potential(self, pos, energy):
        # quadratic potential in y and z
        # double well in x, using quartic potential
        energy[:] = self._u(self._v1, self._v2, self._v3, pos)
    
    def _force(self, pos, force):
        # force is the derivative of the potential
        force[:] = self._f(self._v1, self._v2, self._v3, pos)

    def set_forces(self, timestep):
        state: hoomd.State = self._simulation.state
        with self.cpu_local_force_arrays as arrays:
            with state.cpu_local_snapshot as snapshot:
                pos: hoomd.data.HOOMDArray = snapshot.particles.position
                self._force(pos._coerce_to_ndarray(), arrays.force._coerce_to_ndarray())
                self._potential(pos._coerce_to_ndarray(), arrays.potential_energy._coerce_to_ndarray())
                arrays.virial[:] = np.arange(6)[None, :]

def create_random_orthonormal_basis_3d():
    # generate random orthonormal basis
    while True:
        v1 = np.random.randn(3)
        v1 /= np.linalg.norm(v1)
        v2 = np.random.randn(3)
        v2 -= np.dot(v2, v1) * v1
        v2 /= np.linalg.norm(v2)
        v3 = np.cross(v1, v2)
        if np.linalg.norm(v3) > 1e-3:
            break
    return v1, v2, v3

def rot_mat_between_vecs(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v = np.cross(v1, v2)
    s = np.linalg.norm(v)
    c = np.dot(v1, v2)
    v_x = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
    R = np.eye(3) + v_x + np.dot(v_x, v_x) * (1 - c) / (s ** 2)
    return R

def test_mimse_tilted_sin_large_randomized(simulation_factory):

    N = 100
    # make a random basis for each particle
    vecs = [create_random_orthonormal_basis_3d() for _ in range(N)]
    v1, v2, v3 = zip(*vecs)
    v1 = np.array(v1)
    v2 = np.array(v2)
    v3 = np.array(v3)

    snap = gsd.hoomd.Frame()
    snap.particles.N = N
    snap.particles.position = -0.643501 * v1
    snap.particles.velocity = np.zeros((N, 3))
    snap.particles.types = ['A']
    snap.particles.typeid = [0] * N

    sim = simulation_factory(snap)

    # Setup FIRE sim with Mimse force
    dt = 1e-2
    assert dt <= 1e-2
    fire = hoomd.md.minimize.FIRE(dt, 1e-7, 1.0, 1e-7)

    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    mimse_force = mimse.Mimse(2.0*np.sqrt(N), 2*N)
    tilted_sin_force = TiltedSinVec(v1, v2, v3)
    
    fire.forces.append(mimse_force)
    fire.forces.append(tilted_sin_force)
    fire.methods.append(nve)
    sim.operations.integrator = fire
    
    while not fire.converged:        
        sim.run(10000)

    # Test that the initial snapshot is correct
    bias_pos = -0.643501 * v1
    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        np.testing.assert_array_almost_equal(bias_pos, snap.particles.position, decimal=6)
    mimse_force.push_back(bias_pos)
    mimse_force.kick(0.1 * v1)
    
    fire.reset()
    while not fire.converged:        
        sim.run(100000)
    
    snap = sim.state.get_snapshot()
    compare = 5.639684 * v1
    # print(compare)
    if snap.communicator.rank == 0:
        np.testing.assert_array_almost_equal(snap.particles.position,
                                             compare,
                                             decimal=6)

    bias_pos = 5.639684 * v1
    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        np.testing.assert_array_almost_equal(bias_pos, snap.particles.position, decimal=6)
    mimse_force.push_back(bias_pos)
    mimse_force.kick(0.1 * v1)
    
    fire.reset()
    while not fire.converged:        
        sim.run(100000)
    
    snap = sim.state.get_snapshot()
    compare = 11.9228689 * v1
    if snap.communicator.rank == 0:
        np.testing.assert_array_almost_equal(snap.particles.position,
                                             compare,
                                             decimal=6)
        

def test_mimse_tilted_sin_large(simulation_factory):

    N = 3
    # make a random basis for each particle
    v1 = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]])
    v2 = np.array([[0, 1, 0],
                   [0, 0, 1],
                   [1, 0, 0]])
    v3 = np.array([[0, 0, 1],
                   [1, 0, 0],
                   [0, 1, 0]])

    snap = gsd.hoomd.Frame()
    snap.particles.N = N
    snap.particles.position = -0.643501 * v1
    snap.particles.velocity = np.zeros((N, 3))
    snap.particles.types = ['A']
    snap.particles.typeid = [0] * N

    sim = simulation_factory(snap)

    # Setup FIRE sim with Mimse force
    dt = 1e-2
    assert dt <= 1e-2
    fire = hoomd.md.minimize.FIRE(dt, 1e-7, 1.0, 1e-7)

    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    mimse_force = mimse.Mimse(2.0*np.sqrt(N), 2*N)
    tilted_sin_force = TiltedSinVec(v1, v2, v3)
    
    fire.forces.append(mimse_force)
    fire.forces.append(tilted_sin_force)
    fire.methods.append(nve)
    sim.operations.integrator = fire
    
    while not fire.converged:        
        sim.run(10000)

    # Test that the initial snapshot is correct
    bias_pos = -0.643501 * v1
    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        np.testing.assert_array_almost_equal(bias_pos, snap.particles.position, decimal=6)
    mimse_force.push_back(bias_pos)
    mimse_force.kick(0.1 * v1)
    
    fire.reset()
    while not fire.converged:        
        sim.run(100000)
    
    snap = sim.state.get_snapshot()
    compare = 5.639684 * v1
    # print(compare)
    if snap.communicator.rank == 0:
        np.testing.assert_array_almost_equal(snap.particles.position,
                                             compare,
                                             decimal=6)

    bias_pos = 5.639684 * v1
    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        np.testing.assert_array_almost_equal(bias_pos, snap.particles.position, decimal=6)
    mimse_force.push_back(bias_pos)
    mimse_force.kick(0.1 * v1)
    
    fire.reset()
    while not fire.converged:        
        sim.run(100000)
    
    snap = sim.state.get_snapshot()
    compare = 11.9228689 * v1
    if snap.communicator.rank == 0:
        np.testing.assert_array_almost_equal(snap.particles.position,
                                             compare,
                                             decimal=6)
        
def test_mimse_tilted_sin_large_same(simulation_factory):

    N = 100
    # make a random basis for each particle
    v1 = [np.array([1, 0, 0]) for _ in range(N)]
    v2 = [np.array([0, 1, 0]) for _ in range(N)]
    v3 = [np.array([0, 0, 1]) for _ in range(N)]
    v1 = np.array(v1)
    v2 = np.array(v2)
    v3 = np.array(v3)

    snap = gsd.hoomd.Frame()
    snap.particles.N = N
    snap.particles.position = -0.643501 * v1
    snap.particles.velocity = np.zeros((N, 3))
    snap.particles.types = ['A']
    snap.particles.typeid = [0] * N

    sim = simulation_factory(snap)

    # Setup FIRE sim with Mimse force
    dt = 1e-2
    assert dt <= 1e-2
    fire = hoomd.md.minimize.FIRE(dt, 1e-7, 1.0, 1e-7)

    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    mimse_force = mimse.Mimse(2.0*np.sqrt(N), 2*N)
    tilted_sin_force = TiltedSinVec(v1, v2, v3)
    
    fire.forces.append(mimse_force)
    fire.forces.append(tilted_sin_force)
    fire.methods.append(nve)
    sim.operations.integrator = fire
    
    while not fire.converged:        
        sim.run(10000)

    # Test that the initial snapshot is correct
    bias_pos = -0.643501 * v1
    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        np.testing.assert_array_almost_equal(bias_pos, snap.particles.position, decimal=6)
    mimse_force.push_back(bias_pos)
    mimse_force.kick(0.1 * v1)
    
    fire.reset()
    while not fire.converged:        
        sim.run(100000)
    
    snap = sim.state.get_snapshot()
    compare = 5.639684 * v1
    # print(compare)
    if snap.communicator.rank == 0:
        np.testing.assert_array_almost_equal(snap.particles.position,
                                             compare,
                                             decimal=6)

    bias_pos = 5.639684 * v1
    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        np.testing.assert_array_almost_equal(bias_pos, snap.particles.position, decimal=6)
    mimse_force.push_back(bias_pos)
    mimse_force.kick(0.1 * v1)
    
    fire.reset()
    while not fire.converged:        
        sim.run(100000)
    
    snap = sim.state.get_snapshot()
    compare = 11.9228689 * v1
    if snap.communicator.rank == 0:
        np.testing.assert_array_almost_equal(snap.particles.position,
                                             compare,
                                             decimal=6)