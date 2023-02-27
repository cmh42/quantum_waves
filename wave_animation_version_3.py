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
### Initialise Schrodinger
#######################################

##.*
def initialise_schrodinger(x, psi_x0, V_x, k0=None, hbar=1, m=1, t0=0.0):
    """
    Initialises all the global variables listed below. 
    Inputs x, psi_x0, V_x are list/arrays. 
    Inputs k0, hbar, m, t0 are constants.
    """
    global G_x, G_psi_discrete_x, G_V_x
    global G_hbar, G_m, G_N
    global G_t, G_dt
    global G_dx, G_dk
    global G_k, G_k0
    
    G_x =  np.asarray(x)
    psi_x0 = np.asarray(psi_x0)
    G_V_x = np.asarray(V_x)

    N = G_x.size
    try:
        assert G_x.shape == (N,)
        assert psi_x0.shape == (N,)
        assert G_V_x.shape == (N,)
    except AssertionError as err: 
        print("There was a problem with the array lengths: " + str(err))
        
    G_hbar, G_m, G_N = hbar, m, N
    G_t, G_dt = t0, None
    G_dx = G_x[1] - G_x[0]
    G_dk = 2 * np.pi / (G_N * G_dx)
    G_k0 = -0.5 * G_N * G_dk if k0 is None else k0

    G_k = G_k0 + G_dk * np.arange(G_N)
    G_psi_discrete_x = psi_x0 * np.exp(-1j * G_k[0] * G_x) * G_dx / np.sqrt(2 * np.pi)
    return None 
##.*

##.*
def get_psi_x():
    """
    Returns the non-discrete version of psi_x. 
    """
    global G_psi_discrete_x, G_k, G_x, G_dx
    return G_psi_discrete_x * np.exp(1j * G_k[0] * G_x) * np.sqrt(2 * np.pi) / G_dx

def set_G_dt(dt):
    """
    Updates G_t and the other two  global variables below.
    When first called defines the global variables G_x_evolve_half and G_k_evolve.
    """
    global G_dt, G_x_evolve_half, G_k_evolve
    if dt != G_dt:
        G_dt = dt
        G_x_evolve_half = np.exp(-0.5 * 1j * G_V_x / G_hbar * dt ) 
        G_k_evolve = np.exp(-0.5 * 1j * G_hbar / G_m * (G_k * G_k) * dt)
        return None
    
def time_step(dt, Nsteps = 1):
    """
    Udates G_psi_discrete_k and G_psi_discrete_x using the Fast Fourier 
    Transform and inverse Fast Fourier Transform. Uses the global variables 
    associated with G_dt. Overall update time involved is dt * Nsteps.
    """
    global G_t, G_x_evolve_half, G_k_evolve 
    global G_psi_discrete_k, G_psi_discrete_x

    # Set/update the variables associated with G_dt
    set_G_dt(dt)

    # Then we do the split step fourier method (strang splitting):
    for i in range(Nsteps):
        #half step in position:
        G_psi_discrete_x *= G_x_evolve_half
        #FFT
        G_psi_discrete_k = fft(G_psi_discrete_x)
        #full step in momentum
        G_psi_discrete_k *= G_k_evolve
        #iFFT
        G_psi_discrete_x = ifft(G_psi_discrete_k)
        #half step in position
        G_psi_discrete_x *= G_x_evolve_half

    G_t += dt * Nsteps
    return None 
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
initialise_schrodinger(x=x, psi_x0=psi_x0, V_x=V_x, hbar=hbar, m=m, k0=-28) 
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
    time_step(dt, N_steps)
    psi_x_line.set_data(G_x, 4 * abs(get_psi_x()**2) )
    V_x_line.set_data(G_x, G_V_x)
    # title.set_text('t=%.2f' %S.t)
    return (psi_x_line, V_x_line)

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=16, blit=True)

#anim.save('/content/drive/MyDrive/Summer_Project/wave_experiment.mp4', fps=15, extra_args=['-vcodec', 'libx264'])
# anim.save('wave_experiment.mp4', fps=15, extra_args=['-vcodec', 'libx264'])


plt.show()
##.*
