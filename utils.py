import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition

def participation_ratio1(R):

    C = np.dot(R.T, R)/(R.shape[0]-1)
    eig_vals, eig_vecs = np.linalg.eig(C)
    PR_full = (np.sum(eig_vals.real)**2)/(np.sum(eig_vals.real**2))
    # print(PR_full)
    return PR_full

def participation_ratio2(R1, R2):
    [U1, sig, V1] = np.linalg.svd(R1+R2)
    [U2, noise, V2] = np.linalg.svd(R1-R2)
    sig = sig - noise
    eig_sig_vals = np.square(sig)/(R1.shape[0]-1)
    PR_full = np.sum(eig_sig_vals.real)**2/(np.sum(eig_sig_vals.real**2))
    return PR_full

def var_cutoff(variance, cutoff=0.9):
    variance_explained = np.cumsum(variance)/np.sum(variance)
    dim_cutoff = np.argmax(variance_explained>cutoff) + 1
    return dim_cutoff

def reconstruct_data(R, n_components):
    pca = decomposition.PCA()
    pca.fit(R)
    Rhat = np.dot(pca.transform(R)[:, :n_components], pca.components_[:n_components, :])
    return pca, Rhat

def activity_along_dims(R):
    [U, S, V] = np.linalg.svd(R, full_matrices=True)
    for i in range(15):
        ax = plt.subplot(5,3,i+1)
        ax.plot(R@V[:, i])
        ax.set_ylabel(f'Dim {i+1}')
        # ax.set_ylim([-0.5, 0.5])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()