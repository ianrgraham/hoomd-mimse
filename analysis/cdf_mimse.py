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
distances_all = []
count_A = 0
count_B = 0
count_all = 0
length_A = len(mimse_states_A)
print('length',length_A)

length_B = len(mimse_states_B)
print('length',length_B)


# combine A and B
mimse_states_A = np.array(mimse_states_A)
mimse_states_B = np.array(mimse_states_B)

mimse_states_all = np.concatenate((mimse_states_A,mimse_states_B),axis=0)
length_all = len(mimse_states_all)
print('length',length_all)




for i in range(0,length_A):
    for j in range(0,length_A):
        if (i != j):
            dist = np.linalg.norm(mimse_states_A[i]-mimse_states_A[j])
            distances_A.append(dist)
    count_A += 1
    print('prog_A',np.floor((count_A/length_A)*100))

distances_A = np.array(distances_A)
print('len dist',len(distances_A))
print('max dist',max(distances_A))
print('min dist',min(distances_A))
print('mean dist',np.mean(distances_A))
print('std dist',np.std(distances_A))



for i in range(0,length_B):
    for j in range(0,length_B):
        if (i != j):
            dist = np.linalg.norm(mimse_states_B[i]-mimse_states_B[j])
            distances_B.append(dist)
    count_B += 1
    print('prog_B',np.floor((count_B/length_B)*100))

distances_B = np.array(distances_B)
print('len dist',len(distances_B))
print('max dist',max(distances_B))
print('min dist',min(distances_B))
print('mean dist',np.mean(distances_B))
print('std dist',np.std(distances_B))



for i in range(0,length_all):
    for j in range(0,length_all):
        if (i != j):
            dist = np.linalg.norm(mimse_states_all[i]-mimse_states_all[j])
            distances_all.append(dist)
    count_all += 1
    print('prog_all',np.floor((count_A/length_all)*100))

distances_all = np.array(distances_all)
print('len dist',len(distances_all))
print('max dist',max(distances_all))
print('min dist',min(distances_all))
print('mean dist',np.mean(distances_all))
print('std dist',np.std(distances_all))





        
# print('count',count)

# Concatenate distances from all files
# all_distances = np.concatenate(distances)

# Calculate CDF
sorted_distances_A = np.sort(distances_A)
print('sorted distances',sorted_distances_A)
cdf_A = np.arange(1, len(sorted_distances_A) + 1) / len(sorted_distances_A)

print('len cdf',len(cdf_A))
print('len sorted distances',len(sorted_distances_A))

sorted_distances_B = np.sort(distances_B)
print('sorted distances',sorted_distances_B)
cdf_B = np.arange(1, len(sorted_distances_B) + 1) / len(sorted_distances_B)

print('len cdf',len(cdf_B))
print('len sorted distances',len(sorted_distances_B))

sorted_distances_all = np.sort(distances_all)
print('sorted distances',sorted_distances_all)
cdf_all = np.arange(1, len(sorted_distances_all) + 1) / len(sorted_distances_all)

print('len cdf',len(cdf_all))
print('len sorted distances',len(sorted_distances_all))







# Plot CDF
plt.figure(0)

# x_fit = np.logspace(np.log10(8e0), np.log10(2e1), 100)
# y_fit = x_fit**2.7
# # shift y_fit down
# y_fit = y_fit * 0.0001


# y_fit_3 = np.exp(1.5 * np.log(x_fit) + 2)
# y_fit_4 = np.exp(1.8 * np.log(x_fit) + 3)


# x2_fit = np.linspace(5e-1, 1e1, 100)
# # y_fit = np.exp(2.4 * np.log(x_fit) + 6)
# y2_fit_3 = np.exp(1.5 * np.log(x2_fit) - 5)
# y2_fit_4 = np.exp(1.7 * np.log(x2_fit) - 5)


plt.loglog(sorted_distances_A, cdf_A,label = 'A')
plt.loglog(sorted_distances_B, cdf_B,label = 'B')
plt.loglog(sorted_distances_all, cdf_all,label = 'All')
# # q = slope * np.log(sorted_distances) + (intercept-0.1)
# # plt.loglog(sorted_distances, np.exp(q),label = df0_label)
# # plt.loglog(x_fit, y_fit, label = 'slope = 2.4')
# plt.loglog(x_fit, y_fit, label = 'slope = 2.7')
# plt.loglog(x_fit, y_fit_4, label = 'slope = 1.8')

# plt.loglog(x2_fit, y2_fit_3, label = 'slope = 1.5')
# plt.loglog(x2_fit, y2_fit_4, label = 'slope = 1.7')
plt.ylim([1e-2, 1.4])
plt.xlim([4e0, 2e2])
plt.xlabel('Pairwise Euclidean Distance')
plt.ylabel('CDF')
plt.title(f'CDF')
plt.legend()

plt.savefig(f'new_cdf.png')


plt.figure(1)

plt.hist(distances_A, bins=100, histtype='step', label='A')
plt.hist(distances_B, bins=100, histtype='step', label='B')
plt.hist(distances_all, bins=100, histtype='step', label='All')

plt.legend()
plt.savefig('distances.png')