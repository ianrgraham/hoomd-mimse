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

import init
import sysint



# CONSTANTS
N = 256 ## number of particles
DIM = 3 ## dimension of the system
RANDOM_SEED = 42 ## random seed
NL_PARA = 0.3 ## neighbor list parameter
T_THERMAL = 1.5 ## thermalization temperature
U_SIGMA = 1.5
U_NOUGHT = 20.0
N_ITER = 50


def main():

    cpu = hoomd.device.CPU()
    gpu = hoomd.device.GPU(gpu_id = 1 )
    device = cpu
    print(f"Device: {device}")
    sim: hoomd.Simulation = hoomd.Simulation(device=device)
    sim = hoomd.Simulation(device, seed=RANDOM_SEED)

    rng = init.init_rng(int(RANDOM_SEED)+1)
    L_box = init.L_box_from_rho(N, 1.2, dim=DIM)
    snap = init.approx_euclidean_snapshot(N, L_box, rng, dim=DIM, ratios=[80,20], diams=[1.0, 0.88])
    sim.create_state_from_snapshot(snap)

    cell = hoomd.md.nlist.Cell(NL_PARA)

    # thermalize at T_in
    dt_thermal = 0.002
    integrator = hoomd.md.Integrator(dt=dt_thermal)
    lang = hoomd.md.methods.Langevin(hoomd.filter.All(), T_THERMAL)
    lj = sysint.KA_LJ(cell)
    integrator.forces = [lj]
    integrator.methods = [lang]
    sim.operations.integrator = integrator

    sim.state.thermalize_particle_momenta(hoomd.filter.All(), T_THERMAL)
    print('Start Thermalization\n')
    sim.run(10_000)
    device.notice(f'perf : {sim.tps}')

    # remove all 
    sim.operations.integrator = None
    integrator.forces.pop()
    del integrator, lang

    # quench to IS using FIRE and NVE
    dt_minimizer = 0.002
    fire = hoomd.md.minimize.FIRE(dt_minimizer,
                              force_tol=1e-4,
                              angmom_tol=1e-2,
                              energy_tol=1e-8)
    
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    
    fire.methods.append(nve)
    fire.forces = [lj]
    sim.operations.integrator = fire
    fire.reset()
    while not fire.converged:
        sim.run(100)
    device.notice(f'perf : {sim.tps}')
    print('Quenching complete.')

    # output the energy 
    print(f'Energy: {fire.energy/N}')


    # remove all
    sim.operations.integrator = None
    del fire, nve


    # now lets use MIMSE

    fire = hoomd.md.minimize.FIRE(dt_minimizer,
                              force_tol=1e-4,
                              angmom_tol=1e-2,
                              energy_tol=1e-8)
    
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    fire.methods.append(nve)



    mimse_force = mimse.Mimse(U_SIGMA, U_NOUGHT, subtract_mean=True)

    
    fire.forces.append(mimse_force)

    fire.forces.append(lj)
    sim.operations.integrator = fire

    energies = []
    

    freud_box = freud.box.Box.from_box(sim.state.box)

    ref_pos = sim.state.get_snapshot().particles.position

    print("Start MIMSE \n")
    with tqdm.tqdm(total=N_ITER) as pbar:
        for _ in range(N_ITER):
            # print("")  # to get the progress bar to update without clearning the previous line. remove eventually
            bias_pos = sim.state.get_snapshot().particles.position  # TODO: substitute with on-device copy method
            mimse_force.push_back(bias_pos)
            mimse_force.random_kick(0.05)
            
            post_kick_pos = sim.state.get_snapshot().particles.position
            fire.reset()
            sim.run(1)
            pre_mimse_energy = sim.operations.integrator.forces[0].energy
            pre_mimse_force = sim.operations.integrator.forces[0].forces
            while not fire.converged:
                sim.run(1_000)
            energies.append(sim.operations.integrator.energy)

            # break

            # get biases
            # biases = mimse_force.get_biases()
            # current_pos = sim.state.get_snapshot().particles.position
            pbar.update(1)
            # n_biases = len(biases)
            # bias_distances = [np.linalg.norm(freud_box.wrap(bias - current_pos)) for bias in biases]
            # bias_distances.sort()
            # ref_distance = np.linalg.norm(freud_box.wrap(ref_pos - current_pos))
            # post_kick_distance = np.linalg.norm(freud_box.wrap(post_kick_pos - bias_pos))
            # bias_pos_to_current_pos = np.linalg.norm(freud_box.wrap(bias_pos - current_pos))
            # mimse_energy = sim.operations.integrator.forces[0].energy
            # lj_energy = sim.operations.integrator.forces[1].energy
            # pbar.set_postfix(
            #     {
            #         "n_biases": n_biases,
            #         "largest_bias_distances": bias_distances[-3:],
            #         "ref_distance": ref_distance,
            #         "post_kick_distance": post_kick_distance,
            #         "bias_pos_to_current_pos": bias_pos_to_current_pos,
            #         "pre_mimse_energy": pre_mimse_energy,
            #         "pre_mimse_force": np.linalg.norm(pre_mimse_force),
            #         "mimse_energy": mimse_energy,
            #         "lj_energy": lj_energy,
            #     })
            # mimse_force.prune_biases(3.0)

    plt.plot(energies/N)
    plt.ylabel("Energy")
    plt.xlabel("Iterations")

    plt.savefig(f"md_lj_{N_ITER}_{device}.png")



if __name__ == '__main__':
    main()