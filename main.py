# Python
from datetime import datetime
import time 
import os 
import sys

# MPL
import matplotlib        as mpl
import matplotlib.pyplot as plt

# NumPy
import numpy        as np
import numpy.random as random
import numpy.linalg as linalg
from numpy import cos, exp, log, pi, sin, sqrt

# SciPy
import scipy.fft as fft
import scipy.stats as stats
from scipy.optimize import curve_fit

# FFTW 0.12.0
# There currently exists a bug with conda-forge's builds of FFTW 0.13.0+.
# Version 0.12.0 requires python version 3.10.
import pyfftw

from mpi4py import MPI
from pathlib import Path

from SimulationClass import GravitySimulation

# argv: Np, sim_time, output_level, output_dir, seed
if __name__ == "__main__":
    # Simulation parameters that will be used by all tests.

    # Mp: Mass of each particle. (kg)
    # L: Side length of the (cubical) simulation space.(m)
    # RS: Radius of the (spherical) collapsing body. (m)
    # xc: Center of the body in simulation space. (m)
    # Nc: Number of cells per axis.
    # TODO: Move this to a simulation config file. 
        
    Mp = 0.1
    L = 1
    RS = L/4
    xc = (L/2, L/2, L/2)
    Nc = 128
    
    if len(sys.argv) < 5:
        print("Invalid number of arguments.")
        exit(1)

    Np = int(sys.argv[1])
    sim_time = int(sys.argv[2])
    output_dir = os.path.join(os.getcwd(), os.path.normpath(sys.argv[4]))

    s = None
    if len(sys.argv) >= 6: s = int(sys.argv[5])

    sim = GravitySimulation(Mp, RS, xc, Np, L, Nc, s)

    match int(sys.argv[3]):
        case 0: # Only print to console. 
            sim.evolve_system(sim_time)
       # case 1: # Print to console and output position plots. 
       # case 2: # Print to console, output plots of position, density, and force. 

