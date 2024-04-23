# Python
import os 
import sys

from mpi4py import MPI
from pathlib import Path

import numpy as np
import numpy.random as random
from numpy import cos, sin, pi

#import SimulationMethods as sim

from SimulationMethods import GravitySimulation
from layout import RankLayout

# ----------------------------------------------------------------------------- 
# MPI Message Flags
SENDPARTICLES = 1

# -----------------------------------------------------------------------------
# Uniform probability distribution functions over spherical coordinates. 

def rand_r(N,r_min,r_max,rng=random.default_rng()):
    v_min = 1/3 * r_min**3
    v_max = 1/3 * r_max**3
    v = rng.uniform(v_min,v_max,N)
    return np.cbrt(3*v)

def rand_theta(N,theta_min,theta_max,rng=random.default_rng()):
    u_min = -cos(theta_min)
    u_max = -cos(theta_max)
    u = rng.uniform(u_min,u_max,N)
    return np.arccos(-u)

def rand_phi(N,phi_min,phi_max,rng=random.default_rng()):
    phi = rng.uniform(0,2*pi,N)
    return phi

# -----------------------------------------------------------------------------
# Particle array generation

def init_gbl_particle_array(Np):
    return np.zeros((9,Np), dtype=np.float64)

def init_distr_spherical(params, rng=random.default_rng()):
    Np = int(params["PARTICLES"])
    parts = init_gbl_particle_array(Np)

    r_min = params["R_MIN"]
    r_max = params["R_MAX"]
    theta_min = params["THETA_MIN"]
    theta_max = params["THETA_MAX"]
    phi_min = params["PHI_MIN"]
    phi_max = params["PHI_MAX"]
    rcx = params["CENTER_X"]
    rcy = params["CENTER_Y"]
    rcz = params["CENTER_Z"]
    
    rs = rand_r(Np, r_min, r_max, rng)
    thetas = rand_theta(Np, theta_min, theta_max, rng)
    phis = rand_phi(Np, phi_min, phi_max, rng)

    parts[0] = rs*cos(phis)*sin(thetas) + rcx 
    parts[1] = rs*sin(phis)*sin(thetas) + rcy 
    parts[2] = rs*cos(thetas) + rcz

    return parts

# -----------------------------------------------------------------------------
# Load system parameters and simulation control inputs

def load_params(params_path):
    gbl = {"LENGTH": 1, "CELLS": 128}
    sphere = {"PARTICLES": 32768, 
              "MASS": 0.1,
              "R_MIN": 0,
              "R_MAX": 0.25,
              "THETA_MIN": 0,
              "THETA_MAX": 3.14,
              "PHI_MIN": 0,
              "PHI_MAX": 6.28,
              "CENTER_X": 0.5,
              "CENTER_Y": 0.5,
              "CENTER_Z": 0.5}

    current_obj = gbl

    with open(params_path, 'r') as f:
        for line in f:
            if "[" in line:
                if "]" not in line:
                    print("Error in parameters file: Unclosed object tag.")
                    break
                match line[ line.index("[") + 1 : line.index("]") ].strip():
                    case "Global":
                        current_obj = gbl 
                    case "Sphere":
                        current_obj = sphere
                    case _:
                        print("Error in parameters file: Unsupported object.")
                        break
            elif not line.strip():
                continue 
            else:
                if ":" not in line:
                    print("Error in parameters file: Missing delineator.")
                    break
                key, val = line.split(":")
                key = key.strip()
                val = val.strip()
                
                try:
                    current_obj[key] = float(val)
                except KeyError:
                    print(f"Error in parameters file: {key} not a valid parameter of object {current_obj['Type']}")
                    current_obj = None
                    break
                except ValueError:
                    print(f"Error in parameters file: {key} must be a number.")
                    current_obj = None
                    break

    return gbl, sphere

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

# -----------------------------------------------------------------------------
# Run a time step.

def advance_simulation(sim, sim_time_str, output_dir, verbose_level, debug_flag):
    try:
        sim.evolve_system(float(sim_time_str), output_dir, verbose_level, debug_flag)
        sim_time_str = input("Enter another time to continue simulation, or enter any key to quit\n")
        advance_simulation(sim, sim_time_str, output_dir, verbose_level, debug_flag)
    except ValueError:
        MPI.Finalize()
        exit(0)

# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":
   
    np.set_printoptions(precision=2)

    # ----------------------------------------------------------------
    # Set up parallel stuff.

    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    my_rank = comm.Get_rank()

    # ----------------------------------------------------------------
    # Process Inputs
    
    if my_rank == 0:
        args = validate_inputs()
        if not args: 
            comm.Abort(1)
    else:
        args = None
    
    args = comm.bcast(args, root=0)

    params_dir = args[0]
    output_dir = args[1]
    verbose_level = args[2]
    s = args[3]
    debug_flag = args[4]

    rng = random.default_rng(s)

    if my_rank == 0:
        gbl, sphere = load_params(params_dir)
        if not gbl:
            comm.Abort(1)
        if not sphere:
            comm.Abort(1)
    else:
        sphere = None
    
    sphere = comm.bcast(sphere, root=0)
    Np = int(sphere["PARTICLES"])
    Mp = sphere["MASS"]

    # ----------------------------------------------------------------
    # Generate global particle distribution

    if my_rank == 0:
        parts = init_distr_spherical(sphere, rng)
    lcl_parts = init_gbl_particle_array(Np)

    # ----------------------------------------------------------------
    # Distribute particles to local simulation slices.
         
    if my_rank == 0:
        L = gbl["LENGTH"]
        for i in range(world_size):
            z_min = i * (L/world_size)
            z_max = z_min + (L/world_size)
            mask = np.logical_and(parts[2] > z_min, parts[2] < z_max)
            send_parts = np.where(mask, parts, 0)
            print(send_parts.flags)
            if i == 0:
                lcl_parts = send_parts
            else:
                comm.Send([send_parts,MPI.DOUBLE], dest=i, tag=SENDPARTICLES)
    else:
        comm.Recv([lcl_parts,MPI.DOUBLE], source=0, tag=SENDPARTICLES)

    # ----------------------------------------------------------------
    # Map ranks to the global grid. RankLayout defaults to x-y slabs.

    #if my_rank == 0:
    #    Nc = int(gbl["CELLS"])
    #    try:
    #        layout = RankLayout([Nc,Nc,Nc], world_size)
    #    except ValueError as e:
    #        print(e)
    #        comm.Abort()

    # ----------------------------------------------------------------
    # Begin simulation

    if my_rank == 0:
        sim = GravitySimulation(parts, Np, Mp, L, Nc)

        sim_time_str = input("Simulation successfully initialized. Enter a time to advance the system, or enter any key to quit\n")

        advance_simulation(sim, sim_time_str, output_dir, verbose_level, debug_flag)


