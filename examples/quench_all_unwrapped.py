
import hoomd
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import h5py
import re

import sysint

from os.path import join, isfile
import os
# print(os.getcwd())

workspace_PATH = "../"

import sys
sys.path.insert(1, workspace_PATH)


import numpy as np
import numpy as np

import gsd.hoomd
import freud
import shutil

from natsort import natsorted
from sklearn.linear_model import LinearRegression


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import pickle


N = 256 ## number of particles
DIM = 3 ## dimension of the system
NL_PARA = 0.3 ## neighbor list parameter


output_path = f'{workspace_PATH}/output/'

# postfix_arr = ['A','B','C','D','E','F','G']
# seeds = [99, 44, 111, 121, 131, 141, 151]

postfix_arr = ['D']
seeds = [121]

for i in range(len(postfix_arr)):
    unwrapped_states_path = f'{output_path}/output_states_all_unwrapped_{postfix_arr[i]}'
    quen_states_path = f'{output_path}/quen_states_all_unwrapped_{postfix_arr[i]}'

    # if a dir exists remove contents
    if os.path.exists(quen_states_path):
        shutil.rmtree(quen_states_path)

    if not os.path.exists(quen_states_path):
        os.makedirs(quen_states_path)

    cpu = hoomd.device.CPU()
            # gpu = hoomd.device.GPU(gpu_id = 1 )
    device = cpu
    print(f"Device: {device}")
    sim: hoomd.Simulation = hoomd.Simulation(device=device)
    sim = hoomd.Simulation(device, seed=seeds[i])

    states = os.listdir(unwrapped_states_path)
    print(len(states))
    states = natsorted(states)
    positions_big = np.zeros((int(len(states)), N, DIM))

    cell = hoomd.md.nlist.Cell(NL_PARA)

    lj = sysint.KA_LJ(cell)
    quenched_energy = []
    count = 0
    for file in natsorted(states):
        # do unwrapped states
        cpu = hoomd.device.CPU()
            # gpu = hoomd.device.GPU(gpu_id = 1 )
        device = cpu
        # print(f"Device: {device}")
        sim: hoomd.Simulation = hoomd.Simulation(device=device)
        sim = hoomd.Simulation(device, seed=seeds[i])
        sim.create_state_from_gsd(join(unwrapped_states_path, file),frame=-1)

        # do fire
        fire = hoomd.md.minimize.FIRE(0.002,
                                    force_tol=1e-4,
                                    angmom_tol=1e-2,
                                    energy_tol=1e-8)
        fire.forces = [lj]
        sim.operations.integrator = fire
        
        fire.methods.append(hoomd.md.methods.ConstantVolume(hoomd.filter.All()))
        sim.operations.integrator = fire
        while not fire.converged:
            sim.run(1_000)
        

        quenched_energy.append(lj.energy/N)
        snap = sim.state.get_snapshot()
        posit = snap.particles.position
        image = snap.particles.image
        box = freud.box.Box.from_box(sim.state.box)
        realposition = box.unwrap(posit,image)

        positions_big[count] = realposition


        count += 1

        


        print('count: ',count)
    
    # dump the quenched states and energies
    file_name_pkl = f'quen_states.pkl'
    final_pkl_path = join(quen_states_path,file_name_pkl)
    isExist = os.path.exists(quen_states_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(quen_states_path)
    # Save positions_big using pickle
    with open(final_pkl_path, 'wb') as f:
        pickle.dump(positions_big, f)

    
    file_name_pkl_en = f'quen_en.pkl'
    final_pkl_path_en = join(quen_states_path,file_name_pkl_en)
    with open(final_pkl_path_en, 'wb') as f:
        pickle.dump(quenched_energy, f)
