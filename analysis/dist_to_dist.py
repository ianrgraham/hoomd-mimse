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


file_path_A = f'/home/conor/Documents/Research/hoomd_mimse_Ian/hoomd-mimse/output/quen_states_all_A/quen_A.pkl'
file_path_B = f'/home/conor/Documents/Research/hoomd_mimse_Ian/hoomd-mimse/output/quen_states_all_B/quen_B.pkl'

with open(file_path_A, 'rb') as fi:
    mimse_states_A = pickle.load(fi)

with open(file_path_B, 'rb') as fi:
    mimse_states_B = pickle.load(fi)

distances_A = []
distances_B = []

count_A = 0
count_B = 0

length_A = len(mimse_states_A)
print('length',length_A)

positions_big_A = np.zeros((length_A, 256, 3))

length_B = len(mimse_states_B)
print('length',length_B)

positions_big_B = np.zeros((length_B, 256, 3))


for i in range(0,length_A):
    for j in range(0,length_A):
        if (i != j):
            dist = np.linalg.norm(mimse_states_A[i]-mimse_states_A[j])
            distances_A.append(dist)
    count_A += 1
    print('prog_A',np.floor((count_A/length_A)*100))

distances_A = np.array(distances_A)

for i in range(0,length_B):
    for j in range(0,length_B):
        if (i != j):
            dist = np.linalg.norm(mimse_states_B[i]-mimse_states_B[j])
            distances_B.append(dist)
    count_B += 1
    print('prog_B',np.floor((count_B/length_B)*100))

plt.hist(distances_A, bins=100, histtype='step', label='A')
plt.hist(distances_B, bins=100, histtype='step', label='B')

plt.legend()
plt.savefig('distances.png')