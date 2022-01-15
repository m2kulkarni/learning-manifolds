from turtle import pu
import numpy as np
import matplotlib.pyplot as plt
import plot_utils as putils

class RNN(object):
    """
    Class to implement a general RNN Model

    Parameters
    ---------------
    N = number of parameters
    g = gain constant of the network. g>1.0 is chaotic regime
    p = connection probability
    tau = neuron time constant
    dt = simulation time constant
    N_input = number of input units. 1 for sound, 1 for lickport starts moving
    N_out = number of output units, 1 in our case which drives the lickport
    """
    def __init__(self, N=500, g=1.5, p=0.1, 
                tau=0.1, dt=0.01, N_input=2, 
                N_out=1, T=1):

        self.N = N
        self.g = g
        self.p = p
        self.tau = tau
        self.dt = dt
        self.N_input = N_input
        self.N_out = N_out
        
        # Make the J matrix
        mask = np.random.rand(self.N,self.N)<self.p
        np.fill_diagonal(mask,np.zeros(self.N))
        self.mask = mask
        self.J = self.g / np.sqrt(self.p*self.N) * np.random.randn(self.N,self.N) * mask

        self.W_in = 2*np.random.randn(self.N, self.N_input) - 1
        self.W_out = 2*np.random.randn(self.N_out, self.N) - 1
    
    def step(self, ext):
        self.r = self.r + \
                self.dt/self.tau * \
                (-self.r + np.dot(self.J, self.z) + np.dot(self.W_in, ext.T))

        self.z = np.tanh(self.r)
    
    def add_input(self, I, plot=False):
        self.ext = np.zeros((int(T/dt), self.N_input))
        if I.shape[-1] == 1:
            self.ext[:, 0] = I
        else:
            self.ext = I
        
        plt.plot(self.ext)
        plt.show()


    def simulate(self, T, ext, r0=None):

        time = np.arange(0, T, self.dt)
        time_steps = len(time)

        if r0 is None:
            r0 = 2*np.random.randn(self.N)-1.

        if ext is None:
            ext = np.zeros((time_steps, self.N_input))
        self.ext = ext
        
        self.r = r0
        self.z = np.tanh(self.r)

        #simulation for time_step steps
        record_r = np.zeros((time_steps,self.N))
        record_r[0,:] = self.r
        for i in range(time_steps-1):
            self.step(self.ext[i])
            record_r[i+1, :] = self.r
        
        return self.z, record_r
    
N = 500
g = 1.5
p = 0.1
tau = 0.1
dt = 0.01
N_in = 2
T = 5

def square_wave(amplitude, start, end, T, dt):

    time_steps = int(T/dt)
    wave = np.zeros(time_steps)
    assert(end <= time_steps)
    wave[start:end] = amplitude
    return wave
    
I = square_wave(1, 40, 80, T, dt)
network = RNN(N=N,g=g,p=p,tau=tau,dt=dt,N_input=N_in, T=T)
network.add_input(I)
z_end, r_simulation = network.simulate(T, ext=None)

putils.plot_dynamics(r_simulation)
putils.plot_dynamics_ordered(r_simulation, criteria="max", sort="descending")
