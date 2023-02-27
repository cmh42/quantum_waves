#!/usr/bin/env python
# Code modified from Jake Vanderplas

#######################################
### Initial imports
#######################################

##.*
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.fftpack import fft,ifft
##.*

#######################################
### The Shrodinger class
#######################################

##.*
class Schrodinger:

    def __init__(self, x, psi_x0, V_x, k0 = None, hbar=1, m=1, t0=0.0):
        """
        x = xaxis array of length N giving position
        V_x = yaxis array of length N giving potential
        psi_x0 = array of length N giving intial wave function at t0
        k0 gives the minimumum value of the momentum (there are some constraints on this due to
        the FFT: k0 < k < 2pi/dx where dx = x[1]-x[0])
        default hbar = 1 and mass = 1 and initial time = 0
        """
        self.x, psi_x0, self.V_x = map(np.asarray, (x, psi_x0, V_x))
        N = self.x.size
        assert self.x.shape == (N,)
        assert psi_x0.shape == (N,)
        assert self.V_x.shape == (N,)

        # Set basic attributes        
        self.hbar = hbar
        self.m = m
        self.t = t0
        self.N = len(x)
        # Note that _dt will be accessed and modified via the property self.dt defined below
        self._dt = None

        # Set the positive step (similar to doing dx = b-a/N as coordinates are evenly spaced)
        self.dx = self.x[1] - self.x[0] 

        # Set dk = 2pi/Ndx (do this so FFT looks like continuous fourier transform)
        # This means that dk = 2pi/(b-a)
        self.dk = 2 * np.pi / (self.N * self.dx) 

        # Set k0 according for both cases: input or not        
        if k0 == None:
            self.k0 = -0.5 * self.N * self.dk
        else:
            self.k0 = k0

        # Set k as an array of length N of the form [k0, k0+dk, k0 + 2*dk, ..., k0 + (N-1) * dk]      
        self.k = self.k0 + self.dk * np.arange(self.N) 

        # psi_x is set as a property (with private value  _psi_discrete_x) 
        self.psi_x = psi_x0

        # dt is set as a property (with private value _dt)

    #######################################
    ### Definitions for properties here
    #######################################

    @property
    def psi_x(self):
        """
        The accessor (getter) for ps_x. 
        This just brings it back to the original version of psi(x)
        """
        return (self._psi_discrete_x * np.exp(1j * self.k[0] * self.x) * np.sqrt(2 * np.pi) / self.dx)

    # Because we are using the discrete FFT and iFFT, we have to change psi(x,t) to dx/2pi * psi(x,t) e^(-i k_0 x):
    @psi_x.setter
    def psi_x(self, in_value_psi_x):
        """
        Gives the discrete vesion of psi needed for the FFT and iFFT
        """
        self._psi_discrete_x = (in_value_psi_x * np.exp(-1j * self.k[0] * self.x) * self.dx / np.sqrt(2 * np.pi))
        
    @property
    def dt(self):
        """
        The accessor (getter) for dt. 
        """
        return self._dt

    @dt.setter
    def dt(self, in_value_dt):
        """
        Sets private attributes _dt, _x_evolve_half and _k_evolve
        """
        if in_value_dt != self._dt:
            self._dt = in_value_dt
            self._x_evolve_half = np.exp(-0.5 * 1j * self.V_x / self.hbar * in_value_dt ) 
            self._k_evolve = np.exp(-0.5 * 1j * self.hbar / self.m * (self.k * self.k) * in_value_dt)
        return None

    def time_step(self, dt, Nsteps = 1):
        self.dt = dt

        # Then we do the split step fourier method (strang splitting):
        for i in range(Nsteps):
            #half step in position:
            self._psi_discrete_x *= self._x_evolve_half
            #FFT
            self._psi_discrete_k = fft(self._psi_discrete_x)
            #full step in momentum
            self._psi_discrete_k *= self._k_evolve
            #iFFT
            self._psi_discrete_x = ifft(self._psi_discrete_k)
            #half step in position
            self._psi_discrete_x *= self._x_evolve_half

        self.t += dt * Nsteps
##.*

#######################################
### Gauss
#######################################

##.*

def gauss_x(x, a, x0, k0):
    """
    Inputs: array x, width a, centred at x0, with momentum k0. 
    """
    return ((a * np.sqrt(np.pi)) ** (-0.5) * np.exp(-0.5 * ((x - x0) * 1. / a) ** 2 + 1j * x * k0))

 
##.*

#######################################
### Variables
#######################################

##.*

dt = 0.01
N_steps = 50
#N_steps = 200
t_max = 200

# Change tmax to run for longer / shorter times
frames = int(t_max / float(N_steps * dt))

# Specify constants
hbar = 1.0  
m = 1.0      

# Specify range in x coordinate
N = 2 ** 11
dx = 0.1
x = dx * (np.arange(N) - 0.5 * N)

# This is the potential barrier
V0 = 1.0
L = hbar / np.sqrt(2 * m * V0)
x0 = -60 * L
V_x = np.e**((-x**2)/1) 
#+ np.e**((-(x+75)**2)/1) + np.e**((-(x-75)**2)/1)

# Potential 'walls' at either end
V_x[x < -98] = 1E6 
V_x[x > 98] = 1E6


# Specify initial momentum and quantities derived from it
p0 = np.sqrt(2 * m * 0.8 * V0)
dp2 = p0 * p0 * 1./80
d = hbar / np.sqrt(2 * dp2)

k0 = p0 / hbar
v0 = p0 / m
psi_x0 = gauss_x(x, d, x0, k0)
##.*

#######################################
### Initialise Schrodinger
#######################################

##.*
S = Schrodinger(x=x, psi_x0=psi_x0, V_x=V_x, hbar=hbar, m=m, k0=-28) 
##.*

#######################################
### The animation
#######################################

##.*
fig = plt.figure(figsize=(10,5))

xlim = (-100, 100)
ymin = 0
ymax = 1.5*V0
#was 1.5*V0
ax = fig.add_subplot(111, xlim=xlim, ylim=(ymin * (ymax - ymin), ymax + 0.2 * (ymax - ymin)))
psi_x_line, = ax.plot([], [], c='r', label=r'$|\psi(x)|^2$')
V_x_line, = ax.plot([], [], c='k', label=r'$V(x)$')

title = ax.set_title('')
ax.legend(prop = dict(size=12))
ax.set_xlabel('$x$')
ax.set_ylabel(r'$|\psi(x)|^2$')

def init():
    psi_x_line.set_data([], [])
    V_x_line.set_data([], [])
    return (psi_x_line, V_x_line)

def animate(i):
    S.time_step(dt, N_steps)
    psi_x_line.set_data(S.x, 4 * abs(S.psi_x)**2)
    V_x_line.set_data(S.x, S.V_x)
    # title.set_text('t=%.2f' %S.t)
    return (psi_x_line, V_x_line)

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=16, blit=True)

#anim.save('/content/drive/MyDrive/Summer_Project/wave_experiment.mp4', fps=15, extra_args=['-vcodec', 'libx264'])
# anim.save('wave_experiment.mp4', fps=15, extra_args=['-vcodec', 'libx264'])


plt.show()
##.*
