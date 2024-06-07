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
    N = 1024
    DIM = 3
    rho = 1.2
    ratio = 0.8
    rho_init = 0.1
    L = (N / rho_init) ** (1 / DIM)
    Lz = L
    if DIM == 2:
        Lz = 0
    l = L/2
    snap.particles.N = N
    # set random seed
    np.random.seed(0)
    # snap.particles.position = np.random.rand(N, 3) * l - l / 2
    # put particles on a grid
    n = int(np.ceil(N**(1/DIM)))
    x = np.linspace(-l/2, l/2, n)
    pos = np.array(list(itertools.product(x, repeat=DIM)))
    positions = np.zeros((N, 3))
    positions[:, :DIM] = pos[:N]
    # if DIM == 2:
    #     snap.particles.position[:, 2] = 0
    vel = np.zeros((N, 3))
    # vel[:, :DIM] = np.random.randn(N, DIM)*1.0
    snap.particles.position = positions
    snap.particles.velocity = vel
    snap.particles.types = ['A', 'B']
    typeid = np.ones(N, dtype=int)
    typeid[:int(N*ratio)] = 0
    snap.particles.typeid = typeid
    snap.configuration.box = [L, L, Lz, 0, 0, 0]
    snap.configuration.dimensions = DIM

    sim: hoomd.Simulation = hoomd.Simulation(device=hoomd.device.GPU())
    sim.create_state_from_snapshot(snap)
    # sim.operations.tuners.clear()

    fire = hoomd.md.Integrator(dt=0.005)
    disp_capped = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All(), thermostat=hoomd.md.methods.thermostats.MTTK(1.0, 0.1))
    fire.methods.append(disp_capped)
    add_ka_lj_to_integrator(fire)

    sim.operations.integrator = fire

    sim.run(10_000)

    # Safely quench the system
    fire = hoomd.md.minimize.FIRE(1e-2, 1e-5, 1.0, 1e-5)
    # disp_capped = hoomd.md.methods.DisplacementCapped(hoomd.filter.All(), 0.1)
    disp_capped = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
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
    Lz = L
    if DIM == 2:
        Lz = 0
    final_box = hoomd.Box.from_box([L, L, Lz, 0, 0, 0])
    variant = hoomd.variant.Ramp(0, 1, sim.timestep, 10_000)
    box = hoomd.variant.box.Interpolate(initial_box=initial_box,
                                            final_box=final_box,
                                            variant=variant)

    trigger = hoomd.trigger.Periodic(100)
    box_updater = hoomd.update.BoxResize(trigger=trigger, box=box)
    sim.operations.updaters.append(box_updater)

    sim.run(1)
    
    fire.reset()
    for _ in range(200):
        sim.run(100)
        fire.reset()

    print("final box", sim.state.box)
    print("should be", final_box)

    sim.operations.updaters.clear()

    print("One final quench ...")
    
    while not fire.converged:
        sim.run(1000)

    sim.operations.integrator = None
    del fire

    # Setup FIRE sim with Mimse and LJ forces
    fire = hoomd.md.minimize.FIRE(1e-2, 1e-5, 1.0, 1e-5)
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    # nve = hoomd.md.methods.DisplacementCapped(hoomd.filter.All(), 0.1)
    fire.methods.append(nve)
    
    # mimse
    # mimse_force = mimse.Mimse(1.5*np.sqrt(N), 20.0*N)
    mimse_force = mimse.Mimse(1.5, 20.0, subtract_mean=True)
    fire.forces.append(mimse_force)

    add_ka_lj_to_integrator(fire)
    
    sim.operations.integrator = fire
    # sim.run(0)

    energies = []
    n_iter = 100

    freud_box = freud.box.Box.from_box(sim.state.box)

    ref_pos = sim.state.get_snapshot().particles.position

    print("Running MIMSE ...")
    with tqdm.tqdm(total=n_iter) as pbar:
        for _ in range(n_iter):
            print("")  # to get the progress bar to update without clearning the previous line. remove eventually
            bias_pos = sim.state.get_snapshot().particles.position  # TODO: substitute with on-device copy method
            mimse_force.push_back(bias_pos)
            mimse_force.random_kick(0.01)
            
            post_kick_pos = sim.state.get_snapshot().particles.position
            fire.reset()
            sim.run(1)
            pre_mimse_energy = sim.operations.integrator.forces[0].energy
            pre_mimse_force = sim.operations.integrator.forces[0].forces
            while not fire.converged:
                sim.run(10_000)
            energies.append(sim.operations.integrator.energy)

            # break

            # get biases
            biases = mimse_force.get_biases()
            current_pos = sim.state.get_snapshot().particles.position
            pbar.update(1)
            n_biases = len(biases)
            bias_distances = [np.linalg.norm(freud_box.wrap(bias - current_pos)) for bias in biases]
            bias_distances.sort()
            ref_distance = np.linalg.norm(freud_box.wrap(ref_pos - current_pos))
            post_kick_distance = np.linalg.norm(freud_box.wrap(post_kick_pos - bias_pos))
            bias_pos_to_current_pos = np.linalg.norm(freud_box.wrap(bias_pos - current_pos))
            mimse_energy = sim.operations.integrator.forces[0].energy
            lj_energy = sim.operations.integrator.forces[1].energy
            pbar.set_postfix(
                {
                    "n_biases": n_biases,
                    "largest_bias_distances": bias_distances[-3:],
                    "ref_distance": ref_distance,
                    "post_kick_distance": post_kick_distance,
                    "bias_pos_to_current_pos": bias_pos_to_current_pos,
                    "pre_mimse_energy": pre_mimse_energy,
                    "pre_mimse_force": np.linalg.norm(pre_mimse_force),
                    "mimse_energy": mimse_energy,
                    "lj_energy": lj_energy,
                })
            # mimse_force.prune_biases(3.0)

    plt.plot(energies)
    plt.ylabel("Energy")
    plt.xlabel("Iterations")

    plt.savefig("md_lj.png")

if __name__ == '__main__':
    main()