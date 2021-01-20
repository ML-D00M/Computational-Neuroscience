"""
Computational Neuroscience by University of Washington
(https://www.coursera.org/learn/computational-neuroscience)

Quiz 2 code.
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

import pickle

from Compute STA import Compute STA


FILENAME = 'c1p8.pickle'

with open(FILENAME, 'rb') as f:
    data = pickle.load(f)

stim = data['stim']
rho = data['rho']


# Fill in these values
sampling_period = 2 # in ms
num_timesteps = 150

sta = compute_sta(stim, rho, num_timesteps)

time = (np.arange(-num_timesteps, 0) + 1) * sampling_period

plt.plot(time, sta)
plt.xlabel('Time (ms)')
plt.ylabel('Stimulus')
plt.title('Spike-Triggered Average')

plt.show()
