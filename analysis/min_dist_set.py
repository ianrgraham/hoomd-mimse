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

'''
output_path = f'{workspace_PATH}/output/'

postfix_arr = ['B','C','D']

N = 256 ## number of particles
DIM = 3 ## dimension of the system
for k in range(len(postfix_arr)):
    quen_states_path = f'{output_path}/quen_states_all_{postfix_arr[k]}'
    pkl_state_file = f'quen_states.pkl'

    with open(join(quen_states_path,pkl_state_file), 'rb') as fi:
        globals()[f'mimse_state_{postfix_arr[k]}'] = pickle.load(fi)

    globals()[f'distances_arr_{postfix_arr[k]}'] = []

    globals()[f'mimse_state_{postfix_arr[k]}'] = np.array(globals()[f'mimse_state_{postfix_arr[k]}'])

# for all states in A get min distance to all states in B
# make min_dist_arr same size as A
min_dist_arr = np.ones(len(globals()[f'mimse_state_{postfix_arr[0]}']))
# make every element of min_dist_arr the max value
min_dist_arr = min_dist_arr*np.inf

for i in range(len(globals()[f'mimse_state_{postfix_arr[0]}'])):
    for j in range(len(globals()[f'mimse_state_{postfix_arr[0]}'])):
        if (i != j):
            dist = np.linalg.norm(globals()[f'mimse_state_{postfix_arr[0]}'][i]-globals()[f'mimse_state_{postfix_arr[1]}'][j])
            if dist < min_dist_arr[i]:
                min_dist_arr[i] = dist
    print('prog_B',np.floor((i/len(globals()[f'mimse_state_{postfix_arr[0]}']))*100))

plt.plot(min_dist_arr,label='B vs C')


min_dist_arr = min_dist_arr*np.inf

for i in range(len(globals()[f'mimse_state_{postfix_arr[0]}'])):
    for j in range(len(globals()[f'mimse_state_{postfix_arr[1]}'])):
        if (i != j):
            dist = np.linalg.norm(globals()[f'mimse_state_{postfix_arr[1]}'][i]-globals()[f'mimse_state_{postfix_arr[2]}'][j])
            if dist < min_dist_arr[i]:
                min_dist_arr[i] = dist
    print('prog_B',np.floor((i/len(globals()[f'mimse_state_{postfix_arr[0]}']))*100))

plt.plot(min_dist_arr,label='C vs D')

min_dist_arr = min_dist_arr*np.inf

for i in range(len(globals()[f'mimse_state_{postfix_arr[0]}'])):
    for j in range(len(globals()[f'mimse_state_{postfix_arr[0]}'])):
        if (i != j):
            dist = np.linalg.norm(globals()[f'mimse_state_{postfix_arr[0]}'][i]-globals()[f'mimse_state_{postfix_arr[2]}'][j])
            if dist < min_dist_arr[i]:
                min_dist_arr[i] = dist
    print('prog_B',np.floor((i/len(globals()[f'mimse_state_{postfix_arr[0]}']))*100))


plt.ylabel('min distance')
plt.xlabel('state')
plt.title('min distance between sets')
plt.plot(min_dist_arr,label='B vs D')
plt.legend()





plt.savefig(f'min_distances.png')
'''

output_path = f'{workspace_PATH}/output/'

postfix_arr = ['B','C','D']

N = 256 ## number of particles
DIM = 3 ## dimension of the system

for k in range(len(postfix_arr)):
    quen_states_path = f'{output_path}/quen_states_all_unwrapped_{postfix_arr[k]}'
    pkl_state_file = f'quen_states.pkl'

    with open(join(quen_states_path,pkl_state_file), 'rb') as fi:
        globals()[f'mimse_state_{postfix_arr[k]}'] = pickle.load(fi)

    globals()[f'distances_arr_{postfix_arr[k]}'] = []

    globals()[f'mimse_state_{postfix_arr[k]}'] = np.array(globals()[f'mimse_state_{postfix_arr[k]}'])


# compare min dist across each set

min_dist_arr = np.ones(len(globals()[f'mimse_state_{postfix_arr[0]}']))
min_dist_arr = min_dist_arr*np.inf
for i in range(len(globals()[f'mimse_state_{postfix_arr[0]}'])):
    for j in range(len(globals()[f'mimse_state_{postfix_arr[1]}'])):
            dist = np.linalg.norm(globals()[f'mimse_state_{postfix_arr[0]}'][i]-globals()[f'mimse_state_{postfix_arr[1]}'][j])
            if dist < min_dist_arr[i]:
                min_dist_arr[i] = dist
    print('prog_B',np.floor((i/len(globals()[f'mimse_state_{postfix_arr[0]}']))*100))

plt.loglog(min_dist_arr,label='B vs C')


min_dist_arr = np.ones(len(globals()[f'mimse_state_{postfix_arr[0]}']))
min_dist_arr = min_dist_arr*np.inf
for i in range(len(globals()[f'mimse_state_{postfix_arr[0]}'])):
    for j in range(len(globals()[f'mimse_state_{postfix_arr[2]}'])):
            dist = np.linalg.norm(globals()[f'mimse_state_{postfix_arr[0]}'][i]-globals()[f'mimse_state_{postfix_arr[2]}'][j])
            if dist < min_dist_arr[i]:
                min_dist_arr[i] = dist
    print('prog_B',np.floor((i/len(globals()[f'mimse_state_{postfix_arr[0]}']))*100))

plt.loglog(min_dist_arr,label='B vs D')

min_dist_arr = np.ones(len(globals()[f'mimse_state_{postfix_arr[0]}']))
min_dist_arr = min_dist_arr*np.inf
for i in range(len(globals()[f'mimse_state_{postfix_arr[1]}'])):
    for j in range(len(globals()[f'mimse_state_{postfix_arr[2]}'])):
            dist = np.linalg.norm(globals()[f'mimse_state_{postfix_arr[1]}'][i]-globals()[f'mimse_state_{postfix_arr[2]}'][j])
            if dist < min_dist_arr[i]:
                min_dist_arr[i] = dist
    print('prog_B',np.floor((i/len(globals()[f'mimse_state_{postfix_arr[0]}']))*100))

plt.loglog(min_dist_arr,label='C vs D')
# plot x^1/2 on log log
x = np.linspace(10,1000,1000)
y = x**(1/2)
plt.loglog(x,y,label='x^1/2')
plt.legend()
plt.savefig(f'min_distances.png')


