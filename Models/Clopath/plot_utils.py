import numpy as np
import matplotlib.pyplot as plt

def plot_dynamics(recordings):
    print(recordings.shape) # time_steps x n_neurons
    plt.imshow(recordings.T, aspect="auto", cmap="seismic", origin="lower")
    plt.colorbar()
    plt.show()

def plot_dynamics_ordered(recordings, criteria="mean", sort="ascending"):

    if criteria == "mean":
        arr_val = np.mean(recordings, axis=0)
    elif criteria == "max":
        arr_val = np.max(recordings, axis=0)
    
    arr1inds = arr_val.argsort()
    if sort=="ascending":
        recordings = recordings[:, arr1inds]
    elif sort=="descending":
        recordings = recordings[:, arr1inds[::-1]]

    plot_dynamics(recordings)

        
