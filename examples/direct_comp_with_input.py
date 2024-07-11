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

from os.path import join, isfile
import os
import time

import os
import glob

''''
A was seed = 99
B is seed = 44
C is seed = 111

'''

# CONSTANTS
N = 256 ## number of particles
DIM = 3 ## dimension of the system
# RANDOM_SEED = 111 ## random seed
NL_PARA = 0.3 ## neighbor list parameter
T_THERMAL = 1.5 ## thermalization temperature
U_SIGMA = 1.25
U_NOUGHT = 20.0
N_ITER = 2_500/2
SHIFT = 2_500/2

input_dir = f'/home/conor/Documents/Research/hoomd_mimse_Ian/hoomd-mimse/input'
input_unwrapped_dir = f'{input_dir}/input_unwrapped'
input_wrapped_dir = f'{input_dir}/input_wrapped'

output_dir = f'/home/conor/Documents/Research/hoomd_mimse_Ian/hoomd-mimse/output'
output_unwrapped_dir = f'{output_dir}/output_unwrapped'
output_wrapped_dir = f'{output_dir}/output_wrapped'

base_A_dir = f'base_A'
base_Z_dir = f'base_Z'


# filename = "gsd_SGCOM_mimse_out_all_1250.gsd"
filename = "KA_256_amru.gsd"

postfixes = ['B','C','D','E','F','G']
seeds = [44, 111, 121, 131, 141, 151]



bases_arr = [f'base_A', f'base_Z']


def main():
    for base in bases_arr:
        print(f"Base: {base}")
        for postfix, seed in zip(postfixes, seeds):

            cpu = hoomd.device.CPU()
            # gpu = hoomd.device.GPU(gpu_id = 1 )
            device = cpu
            print(f"Device: {device}")
            sim: hoomd.Simulation = hoomd.Simulation(device=device)
            sim = hoomd.Simulation(device, seed=seed)

            if isfile(filename):
                print(f"Loading from {filename}")
                sim.create_state_from_gsd(filename)


            # make the output directories if they don't exist
            if not os.path.exists(f'{output_wrapped_dir}/{base}/output_states_wrapped_{postfix}'):
                os.makedirs(f'{output_wrapped_dir}/{base}/output_states_wrapped_{postfix}')
            
            if not os.path.exists(f'{output_unwrapped_dir}/{base}/output_states_unwrapped_{postfix}'):
                os.makedirs(f'{output_unwrapped_dir}/{base}/output_states_unwrapped_{postfix}')

            # remove all files in the output directories
            files_all = glob.glob(f'{output_wrapped_dir}/{base}/output_states_wrapped_{postfix}/*.gsd')
            for f in files_all:
                os.remove(f)

            files_all_unwrapped = glob.glob(f'{output_unwrapped_dir}/{base}/output_states_unwrapped_{postfix}/*.gsd')
            for f in files_all_unwrapped:
                os.remove(f)

            cell = hoomd.md.nlist.Cell(NL_PARA)
            lj = sysint.KA_LJ(cell)
        

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
            # fire.reset()
            sim.run(int(seed*2))
            while not fire.converged:
                sim.run(1_000)
            device.notice(f'perf : {sim.tps}')
            print('Quenching complete.')

            print(f'Energy: {fire.energy/N}')


            # remove all
            sim.operations.integrator = None
            del fire, nve


            # now lets use MIMSE

            fire = hoomd.md.minimize.FIRE(dt_minimizer,
                                    force_tol=1e-4,
                                    angmom_tol=1e-2,
                                    energy_tol=1e-4)
            
            nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
            fire.methods.append(nve)



            mimse_force = mimse.Mimse(U_SIGMA, U_NOUGHT, subtract_mean=True)

            
            
            fire.forces.append(mimse_force)

            fire.forces.append(lj)
            sim.operations.integrator = fire

            energies = []
            

            # freud_box = freud.box.Box.from_box(sim.state.box)

            # ref_pos = sim.state.get_snapshot().particles.position

            print("Start MIMSE \n")
            time_start = time.time()
            with tqdm.tqdm(total=N_ITER) as pbar:
                for _ in range(N_ITER):
                    bias_pos = sim.state.get_snapshot().particles.position  # TODO: substitute with on-device copy method
                    mimse_force.push_back(bias_pos)
                    mimse_force.random_kick(0.05)
                    
                    # post_kick_pos = sim.state.get_snapshot().particles.position
                    fire.reset()
                    sim.run(1)
                    # pre_mimse_energy = sim.operations.integrator.forces[0].energy
                    # pre_mimse_force = sim.operations.integrator.forces[0].forces
                    while not fire.converged:
                        sim.run(1_000)
                    energies.append(sim.operations.integrator.energy/N)

                    state_filename_all = f'{output_wrapped_dir}/{base}/output_states_wrapped_{postfix}/gsd_SGCOM_mimse_out_all_wrapped{_+SHIFT}.gsd'
                    hoomd.write.GSD.write(state=sim.state, mode='wb', filename=state_filename_all)

                    
                    final_gsd_path_all = f'{output_unwrapped_dir}/{base}/output_states_unwrapped_{postfix}/gsd_SGCOM_mimse_out_all_unwrapped_{_+SHIFT}.gsd'
                    gsd_all = hoomd.write.GSD(trigger=hoomd.trigger.After(sim.timestep), mode='wb', filename=final_gsd_path_all,dynamic=['property','momentum'],filter=hoomd.filter.All())
                    
                    sim.operations.writers.append(gsd_all)
                    sim.run(1)
                    gsd_all.flush()
                    sim.operations.writers.remove(gsd_all)

                    
                    pbar.update(1)
                    
                    # mimse_force.prune_biases(3.0)
            time_end = time.time()
            print(f"Time taken: {time_end - time_start}")
            plt.semilogx(energies)
            plt.ylabel("Energy")
            plt.xlabel("Iterations")

            plt.savefig(f"md_lj_{N_ITER}_{device}_{postfix}.png")



if __name__ == '__main__':
    main()