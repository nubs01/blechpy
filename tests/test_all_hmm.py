import blechpy
from blechpy.analysis import poissonHMM as phmm

rec_dir = '/media/dsvedberg/Ubuntu Disk/TE_sandbox/DS39_spont_taste_201029_154308'

dat = blechpy.load_dataset(rec_dir)

handler = phmm.HmmHandler(dat)

params = [{'threshold':0.1E-15,'max_iter': 200, 'n_repeats':5, 'time_start':-250,'time_end': 2500, 'dt': 0.01, 'n_states': x,'area': 'GC', 'taste':['Suc', 'NaCl', 'CA', 'QHCl']} for x in [10]]

handler.add_params(params)

handler.run(constraint_func = phmm.sequential_constraint, n_cpu = 6)

handler.plot_saved_models()

import numpy as np
n_states = 5
PI = np.zeros(n_states)
#make a random transition matrix called A with dims n_states x n_states
A = np.random.rand(n_states, n_states)

PI[0] = 1.0
PI[1:] = 0.0
for i in np.arange(n_states):
    if i > 0:  # this part sets the probability of going backwards to 0
        A[i, :i] = 0.0

    if i < n_states - 2:  # this part sets the probability of going to a state more than 1 step ahead to 0
        A[i, i + 2:] = 0.0

    A[i, :] = A[i, :] / np.sum(A[i, :])

A[-1, :] = 0.0
A[-1, -1] = 1.0


n_states = 5
PI = np.zeros(n_states)
A = np.random.rand(n_states, n_states)
n_forward = 3
n_states = len(PI)
PI[0] = 1.0
PI[1:] = 0.0
for i in np.arange(n_states):
    if i > 0:  # this part sets the probability of going backwards to 0
        A[i, :i] = 0.0

    if i < n_states - n_forward:  # this part sets the probability of going to a state more than 1 step ahead to 0
        A[i, i + n_forward:] = 0.0

    A[i, :] = A[i, :] / np.sum(A[i, :])

A[-1, :] = 0.0
A[-1, -1] = 1.0