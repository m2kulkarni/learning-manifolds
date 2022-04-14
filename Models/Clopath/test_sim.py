import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(23213)
N = 50
T = .5
dt = 0.01
tau = 0.1
g = 1.5
p = 0.1
tsteps = int(T//dt)
def square_wave(amplitude, start, end, T, dt):

    time_steps = int(T/dt)
    wave = np.zeros((time_steps, 1))
    assert(end <= time_steps)
    wave[start:end, 0] = amplitude
    return wave

I = square_wave(1, 10, 11, T, dt)

r = np.zeros((tsteps, N))

W = np.random.randn(N, N)*1.1
W = W/np.max(np.linalg.eigvals(W))*0.9
# mask = np.random.rand(N,N)<p
# np.fill_diagonal(mask,np.zeros(N))
# W = g / np.sqrt(p*N) * np.random.randn(N,N)*mask

for i in range(tsteps-1):
    r[i+1,:]= r[i, :]+ dt/tau * (-r[i, :] + np.dot(W, np.tanh(r[i, :])) + (I[i]*np.random.randn(1, N)))

arr1inds = np.sum(abs(r), axis=0).argsort()

# plt.imshow((r), aspect="auto", cmap="Spectral", origin="lower")
# plt.show()
#
# plt.suptitle("mask")
# plt.subplot(211)
# plt.plot((r))
# plt.title("r")
# plt.subplot(212)
# plt.plot(np.tanh(r))
# plt.title("np.tanh(r)")
# plt.ylim([-1, 1])
# plt.show()
#
# plt.plot(np.tanh(r))
# plt.show()
#
alpha = 0.005
alphai = 0.005

##### making the network learn to modulate the cn #######
cn_ind = arr1inds[N//2]
print(cn_ind)
plt.plot(np.tanh(r[:, cn_ind]), label="random sim")
r_cn_i = np.tanh(r[:, cn_ind])

for k in range(200):
    r = np.zeros((tsteps, N))
    W = np.random.randn(N, N)*1.1
    # mask = np.random.rand(N,N)<p
    # np.fill_diagonal(mask,np.zeros(N))
    W = W/np.max(np.linalg.eigvals(W))*0.9

    dw = np.zeros((N, 1))
    for i in range(500):
        r = np.zeros((tsteps, N))
        for i in range(tsteps-1):
            r[i+1,:]= r[i, :]+ dt/tau * (-r[i, :] + np.dot(W, np.tanh(r[i, :])) + (I[i]*np.random.randn(1, N)))
        if i==1:
            r_cn_i = np.tanh(r[:, cn_ind])
        r = np.tanh(r)
        for i in range(N):
            # dw[i]= np.corrcoef(r[:, cn_ind], r[:,i])[1,0]
            dw[i]= np.corrcoef(r[:, cn_ind], r[:,i])[1,0]
        dw_i = dw
        dw = np.dot(dw, dw.T)
        np.fill_diagonal(dw, 0)
        W = W + dw*0.005/np.linalg.norm(dw)
        # inp = inp + di.T*alphai
        # R[:, :, i] = r

    plt.plot(r[:, cn_ind])
plt.legend()
plt.show()
