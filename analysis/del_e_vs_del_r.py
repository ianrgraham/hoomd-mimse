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

import pickle
from natsort import natsorted
from sklearn.linear_model import LinearRegression


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


