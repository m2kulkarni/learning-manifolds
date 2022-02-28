from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_dynamics(recordings, list_cn=None):
    # print(recordings.shape) # time_steps x n_neurons

    fig = plt.figure()
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[1,4])
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(recordings[:,int(list_cn[-1])])
    ax0.set_ylim(-1., 1.)

    ax1 = fig.add_subplot(gs[1, 0])
    im = ax1.imshow(recordings.T, aspect="auto", cmap="Spectral", origin="lower")
    ax1.set_yticks(list_cn)
    s = "Day_id, CN_id"
    # list_ylabels = [s + str(i) + str(x) for (i,x) in enumerate(list_cn)]
    list_ylabels = [f"Day_id={i}, CN_ID={int(list_cn[i])}" for i in range(len(list_cn))]
    ax1.set_yticklabels(labels=list_ylabels)
    # ax1.colorbar()
    # fig.colorbar(im, cax=ax1, orientation='horizontal')

    plt.show()

    # plt.plot(recordings[:, cn])
    # plt.show()

def plot_dynamics_ordered(recordings, criteria="mean", sort="ascending", ḷist_cn=None):

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

    if list_cn is not None:
        list_cn = np.where(arr1inds==list_cn)[0][0]
        plot_dynamics(recordings, list_cn)

    return arr1inds, list_cn

        
