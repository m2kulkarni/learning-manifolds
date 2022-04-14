from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_weight_hist(weight_changes):

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.hist(weight_changes[0].flatten(), bins=100, histtype=u'step', label="initial")
    ax1.hist(weight_changes[len(weight_changes)//2].flatten(), bins=100, histtype=u'step', label="halfway")
    ax1.hist(weight_changes[-1].flatten(), bins=100, histtype=u'step', label="end")
    ax1.legend()

    ax2 = fig.add_subplot(212)
    ax2.hist(np.sum(weight_changes, axis=0).flatten(), bins=100, histtype=u'step', label="weight change during learning")
    ax2.legend()
    plt.show()


def plot_dynamics(list_recordings, list_cn=None):
    # print(recordings.shape) # time_steps x n_neurons

    fig = plt.figure()
    gs = GridSpec(nrows=2, ncols=2, height_ratios=[1,4])
    ax0 = fig.add_subplot(gs[0, :])
    for i in range(len(list_recordings)):
        ax0.plot(list_recordings[i][:,list_cn])
    ax0.set_ylim(-1., 1.)
    ax0.legend(["Before Learning", "End of Day 1"])

    ax1 = fig.add_subplot(gs[1, 0])
    im = ax1.imshow(list_recordings[0].T, aspect="auto", cmap="Spectral", origin="lower")
    ax1.set_yticks(list_cn)
    s = "Day_id, CN_id"
    # list_ylabels = [s + str(i) + str(x) for (i,x) in enumerate(list_cn)]
    list_ylabels = [f"Day_id={i}, CN_ID={int(list_cn[i])}" for i in range(len(list_cn))]
    ax1.set_yticklabels(labels=list_ylabels)
    # ax1.colorbar()
    # fig.colorbar(im, cax=ax1, orientation='horizontal')

    ax2 = fig.add_subplot(gs[1, 1])
    im = ax2.imshow(list_recordings[1].T, aspect="auto", cmap="Spectral", origin="lower")
    ax2.set_yticks(list_cn)
    s = "Day_id, CN_id"
    # list_ylabels = [s + str(i) + str(x) for (i,x) in enumerate(list_cn)]
    list_ylabels = [f"Day_id={i}, CN_ID={int(list_cn[i])}" for i in range(len(list_cn))]
    ax1.set_yticklabels(labels=list_ylabels)


    plt.show()

    # plt.plot(recordings[:, cn])
    # plt.show()

def plot_dynamics_ordered(recordings, criteria="mean", sort="ascending", list_cn=None):

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

    # if list_cn is not None:
    #     list_cn = np.where(arr1inds==list_cn)[0][0]
    #     plot_dynamics(recordings, list_cn)

    return arr1inds, list_cn

        
