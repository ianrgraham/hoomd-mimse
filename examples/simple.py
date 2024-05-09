# Import the plugin module.
import gsd.hoomd
from hoomd.mimse import mimse

# Import the hoomd Python package.
import hoomd
import gsd
from hoomd import operation

import itertools
import pytest
import numpy as np
import matplotlib.pyplot as plt


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

def u_x(x):
    return x**4 - x**2 - 0.5*x

def mimse_u_x(r, sigma, epsilon):
    term = (1 - r*r/(sigma*sigma))
    out = epsilon * term * term
    out[r > sigma] = 0
    return out


def main():
    snap = gsd.hoomd.Frame()
    snap.particles.N = 1
    snap.particles.position = [[-0.5, 0, 0]]
    snap.particles.velocity = [[0, 0, 0]]
    snap.particles.types = ['A']
    snap.particles.typeid = [0]
    snap.configuration.box = [10, 10, 10, 0, 0, 0]
    snap.configuration.dimensions = 3

    sim: hoomd.Simulation = hoomd.Simulation(device=hoomd.device.CPU())
    sim.create_state_from_snapshot(snap)

    # Setup FIRE sim with Mimse force
    fire = hoomd.md.minimize.FIRE(1e-2, 1e-7, 1.0, 1e-7)

    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    mimse_force = mimse.Mimse(1.0, 1.0)
    double_well_force = DoubleWell(x1=-0.5)

    fire.forces.append(mimse_force)
    fire.forces.append(double_well_force)
    fire.methods.append(nve)
    sim.operations.integrator = fire

    sim.run(0)

    bias_pos = np.array([[-0.5, 0, 0]])
    mimse_force.push_back(bias_pos)
    mimse_force.kick(np.array([[0.1, 0, 0]]))

    xs = np.linspace(-1., 1., 100)
    us = u_x(xs)

    mimse_us = mimse_u_x(xs + 0.5, 1.0, 1.0)

    energies = []
    x_pos = []
    fire.reset()
    while not fire.converged:
        sim.run(10)
        sim.run(0)
        energies.append(sim.operations.integrator.forces[0].energy + sim.operations.integrator.forces[1].energy)
        x_pos.append(sim.state.get_snapshot().particles.position[0][0])
    print("FIRE minimization converged after {} steps".format(len(energies)*10))
    plt.plot(xs, mimse_us + us, label="mimse + well")
    plt.plot(xs, us, label="double well")
    plt.plot(x_pos, energies, "o-", label="FIRE")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$U(\vec{r})$")
    plt.legend()
    plt.savefig('double_well.png')
        

if __name__ == '__main__':
    main()