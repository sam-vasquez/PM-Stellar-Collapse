# Python
from datetime import datetime
import time 
import os 
import sys

from mpi4py import MPI
from pathlib import Path

import numpy as np
import numpy.random as random
from numpy import cos, sin, pi

#import SimulationMethods as sim

from SimulationMethods import GravitySimulation

# -----------------------------------------------------------------------------

# Uniform probability distribution functions over spherical coordinates. 
def rand_r(Np,r_min,r_max,rng=random.default_rng()):
    v_min = 1/3 * r_min**3
    v_max = 1/3 * r_max**3
    v = rng.uniform(v_min,v_max,Np)
    return np.cbrt(3*v)

def rand_theta(Np,theta_min,theta_max,rng=random.default_rng()):
    u_min = -cos(theta_min)
    u_max = -cos(theta_max)
    u = rng.uniform(u_min,u_max,Np)
    return np.arccos(-u)

def rand_phi(N,phi_min,phi_max,rng=random.default_rng()):
    phi = rng.uniform(0,2*pi,Np)
    return phi

def init_gbl_particle_array(Np):
    return np.empty((9,Np))

def init_distribution_sphere(parts, Np, xc, RS, rng):
    rs = rand_r(Np, 0, RS, rng)
    thetas = rand_theta(Np, 0, pi, rng)
    phis = rand_phi(Np, 0, 2*pi, rng)

    parts[0] = rs*cos(phis)*sin(thetas) + rc[0]
    parts[1] = rs*sin(phis)*sin(thetas) + rc[1]
    parts[2] = rs*cos(thetas) + rc[2]

def validate_params(params_path):
    params = []
    with open(params_path, 'r') as f:
        for line in f:
            try:
                float(line.split('#')[0].strip())
            except ValueError:
                print(f"Error in parameters file:{line.split('#')[1]} must be a number.")
                params = None
                break
            params += [line.split('#')[0].strip()]
    return params

# argv: config_dir, output_dir, output_level, seed, debug
def validate_inputs():
    if len(sys.argv) < 4:
        print("Usage: python main.py ParamsDir OutputDir VerboseLvl [Seed] [Debug]")
        return None

    params_dir = Path(sys.argv[1]).resolve(strict=False)
    if not os.access(params_dir, os.R_OK):
        print("Permission denied: Unable to read the specified config file.")
        return None
    
    output_dir = Path(sys.argv[2]).resolve(strict=False)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print("Permission denied: Unable to create directory at specified output path.")
        return None
    if not os.access(output_dir, os.W_OK):
        print("Permission denied: Unable to write to the specified output directory.")
        return None

    try:
        verbose_level = int(sys.argv[3])
    except ValueError:
        print("Argument 3 (Verbose level) must be an integer.")
        return None

    s = None
    if len(sys.argv) >= 5: 
        try:
            s = int(sys.argv[4])
        except ValueError:
            print("Argument 4 (RNG Seed) must be an integer.")
            return None

    debug_flag = 0
    if len(sys.argv) >= 6:
        try:
            debug_flag = int(sys.argv[5])
        except ValueError:
            print("Argument 5 (Debug flag) must be an integer.")
            return None

    return [params_dir, output_dir, verbose_level, s, debug_flag]

def advance_simulation(sim, sim_time_str, output_dir, verbose_level, debug_flag):
    try:
        sim.evolve_system(float(sim_time_str), output_dir, verbose_level, debug_flag)
        sim_time_str = input("Enter another time to continue simulation, or enter any key to quit\n")
        advance_simulation(sim, sim_time_str, output_dir, verbose_level, debug_flag)
    except ValueError:
#        MPI.Finalize()
        exit(0)

if __name__ == "__main__":
    # -------------------------------------------------------------
    # Process Inputs
   
    args = validate_inputs()
    if not args: exit(1)
    params_dir = args[0]
    output_dir = args[1]
    verbose_level = args[2]
    s = args[3]
    debug_flag = args[4]


    params = validate_params(params_dir)
    if not params: exit(1)
    Np = int(params[0])
    Mp = float(params[1])
    L = float(params[2])
    Nc = int(params[3])
    r_min = float(params[4])
    r_max = float(params[5])
    theta_min = float(params[6])
    theta_max = float(params[7])
    phi_min = float(params[8])
    phi_max = float(params[9])
    rcx = float(params[10])
    rcy = float(params[11])
    rcz = float(params[12])
    rc = (rcx, rcy, rcz)

    RS = L/4

    # -------------------------------------------------------------
    # Set up parallel stuff.

#    comm = MPI.COMM_WORLD
#    world_size = comm.Get_size()
#    my_rank = comm.Get_rank()

    # -------------------------------------------------------------
    # Begin Simulation

    sim = GravitySimulation(Mp, RS, rc, Np, L, Nc, s)

    sim_time_str = input("Simulation successfully initialized. Enter a time to advance the system, or enter any key to quit\n")

    advance_simulation(sim, sim_time_str, output_dir, verbose_level, debug_flag)


