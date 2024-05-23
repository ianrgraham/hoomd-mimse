# Import the plugin module.
import gsd.hoomd
from hoomd.mimse import mimse

# Import the hoomd Python package.
import hoomd
import gsd
import freud
from hoomd import operation

import itertools
import pytest
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def add_ka_lj_to_integrator(integrator, nlist=None):
    if nlist is None:
        nlist = hoomd.md.nlist.Cell(0.3)
    ka = hoomd.md.pair.LJ(nlist=nlist)
    ka.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)
    ka.params[('A', 'B')] = dict(epsilon=1.5, sigma=0.8)
    ka.params[('B', 'B')] = dict(epsilon=0.5, sigma=0.88)
    ka.r_cut[('A', 'A')] = 2.5
    ka.r_cut[('A', 'B')] = 2.5
    ka.r_cut[('B', 'B')] = 2.5
    integrator.forces.append(ka)


def main():

    # Setup random simulation
    snap = gsd.hoomd.Frame()
    N = 256
    rho = 1.2
    rho_init = 0.1
    L = (N / rho_init) ** (1 / 3)
    l = L/2
    snap.particles.N = N
    # set random seed
    np.random.seed(0)
    snap.particles.position = np.random.rand(N, 3) * l - l / 2
    snap.particles.velocity = np.zeros((N, 3))
    snap.particles.types = ['A', 'B']
    typeid = np.ones(N, dtype=int)
    typeid[:int(N*0.8)] = 0
    snap.particles.typeid = typeid
    snap.configuration.box = [L, L, L, 0, 0, 0]
    snap.configuration.dimensions = 3

    sim: hoomd.Simulation = hoomd.Simulation(device=hoomd.device.GPU())
    sim.create_state_from_snapshot(snap)

    # Safely quench the system
    fire = hoomd.md.minimize.FIRE(1e-2, 1e-10, 1.0, 1e-10)
    disp_capped = hoomd.md.methods.DisplacementCapped(hoomd.filter.All(), 0.1)
    fire.methods.append(disp_capped)

    # Add LJ force
    add_ka_lj_to_integrator(fire)

    sim.operations.integrator = fire

    sim.run(1)

    print("Quenching the system ...")

    with tqdm.tqdm() as pbar:
        force = sim.operations.integrator.forces[0].forces
        max_force = np.max(np.linalg.norm(force, axis=1))
        pbar.set_postfix({"energy": sim.operations.integrator.energy, "max_force": max_force})
        while not fire.converged:
            sim.run(10_000)
            pbar.update(1)
            force = sim.operations.integrator.forces[0].forces
            max_force = np.max(np.linalg.norm(force, axis=1))
            pbar.set_postfix({"energy": sim.operations.integrator.energy, "max_force": max_force})

    print("Compressing the box ...")

    # Compress the box
    initial_box = sim.state.box
    print(initial_box)
    L = (N / rho) ** (1 / 3)
    final_box = hoomd.Box.from_box([L, L, L, 0, 0, 0])
    variant = hoomd.variant.Ramp(0, 1, sim.timestep, 10_000)
    box = hoomd.variant.box.Interpolate(initial_box=initial_box,
                                            final_box=final_box,
                                            variant=variant)

    trigger = hoomd.trigger.Periodic(100)
    box_updater = hoomd.update.BoxResize(trigger=trigger, box=box)
    sim.operations.updaters.append(box_updater)

    sim.run(1)
    
    fire.reset()
    for _ in range(100):
        sim.run(100)
        fire.reset()

    print("final box", sim.state.box)

    print("One final quench ...")
    
    while not fire.converged:
        sim.run(1000)

    # Setup FIRE sim with Mimse and LJ forces
    fire = hoomd.md.minimize.FIRE(1e-2, 1e-10, 1.0, 1e-10)
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    fire.methods.append(nve)
    
    # mimse
    mimse_force = mimse.Mimse(1.0, 100.0)
    fire.forces.append(mimse_force)

    add_ka_lj_to_integrator(fire)
    
    sim.operations.integrator = fire

    energies = []
    n_iter = 20

    print("Running MIMSE ...")
    with tqdm.tqdm(total=n_iter) as pbar:
        for _ in range(n_iter):
            print("")
            bias_pos = sim.state.get_snapshot().particles.position  # TODO: substitute with on-device copy method
            mimse_force.push_back(bias_pos)
            mimse_force.random_kick(1.0)
            fire.reset()
            while not fire.converged:
                sim.run(1000)
            energies.append(sim.operations.integrator.energy)

            # get biases
            biases = mimse_force.get_biases()
            current_pos = sim.state.get_snapshot().particles.position
            pbar.update(1)
            n_biases = len(biases)
            freud_box = freud.box.Box.from_box(sim.state.box)
            bias_distances = [np.linalg.norm(freud_box.wrap(bias - current_pos)) for bias in biases]
            bias_distances.sort()
            pbar.set_postfix({"n_biases": n_biases, "largest_bias_distances": bias_distances[-3:]})
            # mimse_force.prune_biases(2.0)

    plt.plot(energies)
    plt.ylabel("Energy")
    plt.xlabel("Iterations")

    plt.savefig("md_lj.png")

if __name__ == '__main__':
    main()