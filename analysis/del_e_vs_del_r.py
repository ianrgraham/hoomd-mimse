import hoomd
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import h5py
import re

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

import pickle
from natsort import natsorted
from sklearn.linear_model import LinearRegression

'''
file_path_en = f'/home/conor/Documents/Research/hoomd_mimse_Ian/hoomd-mimse/output/quen_states_all_A/quen_en.pkl'
file_path_states = f'/home/conor/Documents/Research/hoomd_mimse_Ian/hoomd-mimse/output/quen_states_all_A/quen_states.pkl'
# file_path_B = f'/home/conor/Documents/Research/hoomd_mimse_Ian/hoomd-mimse/output/quen_states_all_B/quen_B.pkl'



with open(file_path_states, 'rb') as fi:
    mimse_states_A = pickle.load(fi)

with open(file_path_en, 'rb') as fi:
    mimse_en_A = pickle.load(fi)

# with open(file_path_B, 'rb') as fi:
#     mimse_states_B = pickle.load(fi)

distances_A = []
energies_A = []

count_A = 0
count_B = 0

length_A = len(mimse_states_A)
print('length',length_A)

positions_big_A = np.zeros((length_A, 256, 3))


ref_state_A = mimse_states_A[0]
ref_en_A = mimse_en_A[0]


for i in range(0,length_A):
    dist = np.linalg.norm(mimse_states_A[i]-ref_state_A)
    
    distances_A.append(dist)
    count_A += 1
    print('prog_A',np.floor((count_A/length_A)*100))

plt.plot(distances_A, mimse_en_A, 'o')
plt.xlabel('Distance')
plt.ylabel('Energy')
plt.savefig('del_e_vs_del_r.png')
'''


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
    # get path dirs for states and energies
    quen_states_path = f'{output_path}/quen_states_all_{postfix_arr[k]}'
    pkl_state_file = f'quen_states.pkl'

    quen_en_path = f'{output_path}/quen_states_all_{postfix_arr[k]}'
    pkl_en_file = f'quen_en.pkl'

    # load using pickle
    with open(join(quen_states_path,pkl_state_file), 'rb') as fi:
        globals()[f'mimse_state_{postfix_arr[k]}'] = pickle.load(fi)

    with open(join(quen_en_path,pkl_en_file), 'rb') as fi:
        globals()[f'mimse_en_{postfix_arr[k]}'] = pickle.load(fi)

    # set empty array for distances and energies
    globals()[f'distances_arr_{postfix_arr[k]}'] = []
    globals()[f'energies_arr_{postfix_arr[k]}'] = []


    # make loaded data into numpy arrays
    globals()[f'mimse_state_{postfix_arr[k]}'] = np.array(globals()[f'mimse_state_{postfix_arr[k]}'])
    globals()[f'mimse_en_{postfix_arr[k]}'] = np.array(globals()[f'mimse_en_{postfix_arr[k]}'])

    # set count to 0
    globals()[f'count_{postfix_arr[k]}'] = 0

    # set up empty array for positions | size: (length of states, N, DIM)
    globals()[f'positions_big_{postfix_arr[k]}'] = np.zeros((len(globals()[f'mimse_state_{postfix_arr[k]}']), N, DIM))

    # set reference state and energy(first state and energy)
    ref_state = globals()[f'mimse_state_{postfix_arr[k]}'][0]
    ref_en = globals()[f'mimse_en_{postfix_arr[k]}'][0]

    for i in range(len(globals()[f'mimse_state_{postfix_arr[k]}'])):

        # calculate distance and append to distances array
        dist = np.linalg.norm(freud_box.wrap(globals()[f'mimse_state_{postfix_arr[k]}'][i]-ref_state))
        globals()[f'distances_arr_{postfix_arr[k]}'].append(dist)

        globals()[f'count_{postfix_arr[k]}'] += 1
        print(f'prog_{postfix_arr[k]}',np.floor((globals()[f'count_{postfix_arr[k]}']/len(globals()[f'mimse_state_{postfix_arr[k]}']))*100))

    plt.figure(0)
    plt.plot(globals()[f'distances_arr_{postfix_arr[k]}'], globals()[f'mimse_en_{postfix_arr[k]}'], 'o', label = f'{postfix_arr[k]}')
    plt.xlabel('Distance')
    plt.ylabel('Energy')
    plt.legend()
    

plt.savefig(f'del_e_vs_del_r.png')
    



