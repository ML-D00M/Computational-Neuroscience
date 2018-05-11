# computational_neuroscience

"""
Computational neuroscience by University of Washington

Code to compute spike-triggered average of H1 neuron
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def compute_sta(stim, rho, num_timesteps):
    """Compute the spike-triggered average from a stimulus and spike-train.
    
    Args:
        - stim: stimulus time-series. This variable is 600000 long 
        and represents all the stimulus; a 300 ms window before 
        the neuron spikes represents the stimulus responsible 
        for the firing of that spike.
        
        - rho: spike-train time-series. This variable representing when 
        a neuron is firing with "1" and not firing with "0" is 600000 long, 
        and includes a little over 50000 firing of a neuron.
        
        - num_timesteps: how many timesteps to use in STA
        
    Returns:
        spike-triggered average for num_timesteps timesteps before spike"""
    
    sta = np.zeros((num_timesteps,))

    # This command finds the indices of all of the spikes that occur
    # after 300 ms into the recording.
    spike_times = rho[num_timesteps:].nonzero()[0] + num_timesteps

    print(spike_times)

    # Fill in this value. Note that you should not count spikes that occur
    # before 300 ms into the recording.
    num_spikes = 30
    
    # Compute the spike-triggered average of the spikes found.
    # To do this, compute the average of all of the vectors
    # starting 300 ms (exclusive) before a spike and ending at the time of
    # the event (inclusive). Each of these vectors defines a list of
    # samples that is contained within a window of 300 ms before each
    # spike. The average of these vectors should be completed in an
    # element-wise manner.
    # 
    # Your code goes here.
    

    for i in np.nditer(spike_times):

        if i > 300:

            num_spikes += 1

    print(num_spikes)


    for i in range(num_spikes):

        window = stim[spike_times[i] - num_timesteps + 1: spike_times[i] + 1]

        sta = sta + window

    sta = sta/num_spikes

    return sta
