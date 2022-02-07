from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_dynamics(recordings, cn=None):
    # print(recordings.shape) # time_steps x n_neurons

    fig = plt.figure()
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[1,4])
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(recordings[:, cn])
    ax0.set_ylim(-1., 1.)

    ax1 = fig.add_subplot(gs[1, 0])
    im = ax1.imshow(recordings.T, aspect="auto", cmap="Spectral", origin="lower")
    ax1.set_yticks([cn])
    ax1.set_yticklabels(labels=[f"CN_ID={cn}"])
    # ax1.colorbar()
    # fig.colorbar(im, cax=ax1, orientation='horizontal')

    plt.show()

    # plt.plot(recordings[:, cn])
    # plt.show()

def plot_dynamics_ordered(recordings, criteria="mean", sort="ascending", cn=None):

    if criteria == "mean":
        arr_val = np.mean(recordings, axis=0)
    elif criteria == "max":
        arr_val = np.max(recordings, axis=0)
    elif criteria == "max_initial":
        arr_val = np.max(recordings[:100, :], axis=0)

    
    arr1inds = arr_val.argsort()
    if sort=="ascending":
        recordings = recordings[:, arr1inds]
    elif sort=="descending":
        arr1inds = arr1inds[::-1]
        recordings = recordings[:, arr1inds]

    if cn is not None:
        cn = np.where(arr1inds==cn)[0][0]
        plot_dynamics(recordings, cn)

    return arr1inds,cn

        
