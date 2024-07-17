import hoomd
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import h5py
import re
import freud

from os.path import join, isfile
import os
# print(os.getcwd())

workspace_PATH = "../"

import sys
sys.path.insert(1, workspace_PATH)


from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()

rank = comm.Get_rank()

import gsd.hoomd

import pickle
from natsort import natsorted
from sklearn.linear_model import LinearRegression

U_SIGMA = 1.25

if rank == 0:
    print('Rank 0')
    cpu = hoomd.device.CPU()
    device = cpu
    print(f"Device: {device}")
    sim: hoomd.Simulation = hoomd.Simulation(device=device)
    sim = hoomd.Simulation(device, seed=42)
    sim.create_state_from_gsd(f'/home/conor/Documents/Research/hoomd_mimse_Ian/hoomd-mimse/examples/gsd_SGCOM_mimse_out_all_unwrapped_1250_Z.gsd')
    freud_box = freud.box.Box.from_box(sim.state.box)


output_path = f'{workspace_PATH}/output_Z/'

postfix_arr = ['B','C','D','E','F','G']

N = 256 ## number of particles
DIM = 3 ## dimension of the system

for k in range(len(postfix_arr)):
    quen_states_path = f'{output_path}/quen_states_all_{postfix_arr[k]}'
    pkl_state_file = f'quen_states.pkl'

    with open(join(quen_states_path,pkl_state_file), 'rb') as fi:
        globals()[f'mimse_state_{postfix_arr[k]}'] = pickle.load(fi)

    globals()[f'distances_arr_{postfix_arr[k]}'] = []

    globals()[f'mimse_state_{postfix_arr[k]}'] = np.array(globals()[f'mimse_state_{postfix_arr[k]}'])


# compare min dist across each set


     

'''

min_dist_arr = np.ones(len(globals()[f'mimse_state_{postfix_arr[0]}']))
min_dist_arr = min_dist_arr*np.inf
for i in range(len(globals()[f'mimse_state_{postfix_arr[0]}'])):
    for j in range(len(globals()[f'mimse_state_{postfix_arr[1]}'])):
            dist = np.linalg.norm(freud_box.wrap(globals()[f'mimse_state_{postfix_arr[0]}'][i]-globals()[f'mimse_state_{postfix_arr[1]}'][j]))
            if dist < min_dist_arr[i]:
                min_dist_arr[i] = dist
    print('prog_B',np.floor((i/len(globals()[f'mimse_state_{postfix_arr[0]}']))*100))

plt.loglog(min_dist_arr,label='B vs C')


min_dist_arr = np.ones(len(globals()[f'mimse_state_{postfix_arr[0]}']))
min_dist_arr = min_dist_arr*np.inf
for i in range(len(globals()[f'mimse_state_{postfix_arr[0]}'])):
    for j in range(len(globals()[f'mimse_state_{postfix_arr[2]}'])):
            dist = np.linalg.norm(freud_box.wrap(globals()[f'mimse_state_{postfix_arr[0]}'][i]-globals()[f'mimse_state_{postfix_arr[2]}'][j]))
            if dist < min_dist_arr[i]:
                min_dist_arr[i] = dist
    print('prog_C',np.floor((i/len(globals()[f'mimse_state_{postfix_arr[0]}']))*100))

plt.loglog(min_dist_arr,label='B vs D')

min_dist_arr = np.ones(len(globals()[f'mimse_state_{postfix_arr[0]}']))
min_dist_arr = min_dist_arr*np.inf
for i in range(len(globals()[f'mimse_state_{postfix_arr[0]}'])):
    for j in range(len(globals()[f'mimse_state_{postfix_arr[3]}'])):
            dist = np.linalg.norm(freud_box.wrap(globals()[f'mimse_state_{postfix_arr[0]}'][i]-globals()[f'mimse_state_{postfix_arr[3]}'][j]))
            if dist < min_dist_arr[i]:
                min_dist_arr[i] = dist
    print('prog_D',np.floor((i/len(globals()[f'mimse_state_{postfix_arr[0]}']))*100))

plt.loglog(min_dist_arr,label='B vs E')

min_dist_arr = np.ones(len(globals()[f'mimse_state_{postfix_arr[0]}']))
min_dist_arr = min_dist_arr*np.inf
for i in range(len(globals()[f'mimse_state_{postfix_arr[0]}'])):
    for j in range(len(globals()[f'mimse_state_{postfix_arr[4]}'])):
            dist = np.linalg.norm(freud_box.wrap(globals()[f'mimse_state_{postfix_arr[0]}'][i]-globals()[f'mimse_state_{postfix_arr[4]}'][j]))
            if dist < min_dist_arr[i]:
                min_dist_arr[i] = dist
    print('prog_E',np.floor((i/len(globals()[f'mimse_state_{postfix_arr[0]}']))*100))

plt.loglog(min_dist_arr,label='B vs F')

min_dist_arr = np.ones(len(globals()[f'mimse_state_{postfix_arr[0]}']))
min_dist_arr = min_dist_arr*np.inf
for i in range(len(globals()[f'mimse_state_{postfix_arr[0]}'])):
    for j in range(len(globals()[f'mimse_state_{postfix_arr[5]}'])):
            dist = np.linalg.norm(freud_box.wrap(globals()[f'mimse_state_{postfix_arr[0]}'][i]-globals()[f'mimse_state_{postfix_arr[5]}'][j]))
            if dist < min_dist_arr[i]:
                min_dist_arr[i] = dist
    print('prog_F',np.floor((i/len(globals()[f'mimse_state_{postfix_arr[0]}']))*100))

plt.loglog(min_dist_arr,label='B vs G')



min_dist_arr = np.ones(len(globals()[f'mimse_state_{postfix_arr[0]}']))
min_dist_arr = min_dist_arr*np.inf
for i in range(len(globals()[f'mimse_state_{postfix_arr[1]}'])):
    for j in range(len(globals()[f'mimse_state_{postfix_arr[2]}'])):
            dist = np.linalg.norm(freud_box.wrap(globals()[f'mimse_state_{postfix_arr[1]}'][i]-globals()[f'mimse_state_{postfix_arr[2]}'][j]))
            if dist < min_dist_arr[i]:
                min_dist_arr[i] = dist
    print('prog_G',np.floor((i/len(globals()[f'mimse_state_{postfix_arr[0]}']))*100))

plt.loglog(min_dist_arr,label='C vs D')

'''


for postfix in postfix_arr:
    fig_count = 0
    base = globals()[f'mimse_state_{postfix}']
    min_dist_arr = np.ones(len(base))
    min_dist_arr = min_dist_arr*np.inf
    tmp_postfix_arr = postfix_arr.copy()
    tmp_postfix_arr.remove(postfix)
    plt.figure(fig_count)
    fig_count += 1
    for postfix2 in tmp_postfix_arr:
        for i in range(len(base)):
        
            for j in range(len(globals()[f'mimse_state_{postfix2}'])):

                dist = np.linalg.norm(freud_box.wrap(base[i]-globals()[f'mimse_state_{postfix2}'][j]))
                if dist < min_dist_arr[i]:
                    min_dist_arr[i] = dist
                print(f'{postfix}_prog_{postfix2}',((i/(len(globals()[f'mimse_state_{postfix2}'])))*100))
        plt.loglog(min_dist_arr,label=f'{postfix} vs {postfix2}')
    x = np.linspace(10,3000,100)
    y = x**(1/2)
    plt.loglog(x,y,label='x^1/2',linestyle='--')
    plt.axhline(y=2*U_SIGMA, color='r', linestyle='--',label='2*U_SIGMA')
    plt.legend()
    plt.savefig(f'min_distances_base_{postfix}.png')








# plot x^1/2 on log log



