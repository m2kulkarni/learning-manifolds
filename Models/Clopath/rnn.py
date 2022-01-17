from time import time_ns
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
        self.W_fb = 2*np.random.randn(self.N, 1) - 1

    
    def step(self, ext):

        # print(f"{np.dot(self.J, self.z).shape}, {np.dot(self.W_in, ext.T).shape}")
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
        
        if plot:
            plt.plot(self.ext)
            plt.show()
        return self.ext

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
            # print(ext[i].shape)
            self.step(self.ext[i])
            record_r[i+1, :] = self.r
        
        return self.z, record_r
    
    def initialize_cursor(self, cursor_distance_initial):
        """
        cursor == lickport
        everything in m/s

        cursor_velocity must be dependent on CN activity but right now we just let it be constant
        """
        self.cursor_velocity = 0.05
        self.cursor_distance = cursor_distance_initial
        self.cursor_distance_initial = cursor_distance_initial
    
    def learning(self, T, ext, conditioned_neuron, r0=None, day_id=None):

        self.conditioned_neuron =  conditioned_neuron
        self.current_day_id = day_id
        self.initialize_cursor(1)
        time_steps = int(T/self.dt)
        self.P = np.eye(self.N, self.N)*0.05

        if r0 is None:
            r0 = 2*np.random.randn(self.N)-1.
        if ext is None:
            ext = np.zeros((time_steps, self.N_input))
        self.ext = ext

        self.r = r0 # remember to give previous trial r0 to the network
        self.z = np.tanh(self.r)
        
        record_r = np.zeros((time_steps, self.N))
        record_r[0,:] = self.r

        for i in range(time_steps-1):
            """
            abcdefghijklmnopqrstuvwxyz
            """
            error_val = self.cursor_distance*0.1 # error = a*how far the lickport is + b*how fast is it moving. have to figure out a,b
            # print(error_val)
            # print(self.W_fb.shape)
            self.error = self.W_fb*error_val

            if i%2 == 0:
                Pr = np.dot(self.P, self.r)
                self.P -= np.outer(Pr, self.r).dot(self.P)/(1+np.dot(self.r, Pr))
                self.e_minus = self.error
                self.dw = np.outer(np.dot(self.P, self.r), self.e_minus)
                self.J -= self.dw
            
            self.step(ext[i])
            record_r[i+1, :] = self.r

            if self.z[self.conditioned_neuron] >= 0.3:
                self.cursor_distance -= self.cursor_velocity

        return record_r, np.tanh(record_r)

    def participation_ratio(self, eig_vals):
        return (np.sum(eig_vals.real)**2)/(np.sum(eig_vals.real**2))

    def calculate_manifold(self, T, trials, I, pulse_end):

        time_steps = I.shape[0]
        ext = np.zeros((time_steps, self.N_input))
        ext[:, 0] = I

        npoints = time_steps-pulse_end
        activity = np.zeros((trials*npoints,self.N))
        
        for i in range(trials):
            z_end, r_simulation = network.simulate(T, ext=ext)
            z_simulation = np.tanh(r_simulation)
            activity[i*npoints:(i+1)*npoints, :] = z_simulation[pulse_end:, :]
            print(f"{i+1} completed")

        print(f"time_steps={time_steps}, npoints={npoints}, trials={trials}, activity.shape={activity.shape}")

        cov = np.cov(activity.T)
        eig_val, eig_vec = np.linalg.eig(cov)
        pr = self.participation_ratio(eig_val)
        activity_manifold = activity @ eig_vec

        return activity_manifold, activity, eig_val, eig_vec, pr, cov
    
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

pulse_amplitude = 1
pulse_start = 10    
pulse_end = 30
pulse_length = pulse_end-pulse_start
I = square_wave(pulse_amplitude, pulse_start, pulse_end, T, dt)

network = RNN(N=N,g=g,p=p,tau=tau,dt=dt,N_input=N_in, T=T)
network.add_input(I)
z_end, r_simulation = network.simulate(T, ext=None)
activity_manifold, activity, eig_val, eig_vec, pr, cov = network.calculate_manifold(T, 10, I, pulse_end=pulse_end)

cn = np.random.choice(np.max(r_simulation[:100, :], axis=0).argsort()[:10])

print(cn)
# print(r_simulation[, :])
r_learn, z_learn = network.learning(T, None, conditioned_neuron=cn, r0=r_simulation[-1], day_id=0)


print(f"Participation Ratio: {pr}")
print(np.where(np.cumsum(eig_val.real)/np.sum(eig_val.real)>0.9)[0][0])
# plt.plot(np.cumsum(eig_val.real)/np.sum(eig_val.real))
# plt.show()

putils.plot_dynamics(np.tanh(r_simulation), cn=cn)
putils.plot_dynamics_ordered(np.tanh(r_simulation), criteria="max_initial", sort="ascending", cn=cn)

putils.plot_dynamics(np.tanh(r_learn), cn=cn)
putils.plot_dynamics_ordered(np.tanh(r_learn), criteria="max_initial", sort="ascending", cn=cn)



"""
simulate → calculate manifold → feedback learning with cursor velocity → simulate → calculate manifold →   
day 1 complete → repeat for day 2 with different conditioned neuron
"""