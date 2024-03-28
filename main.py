# Python
from datetime import datetime
import time 
import os 
import sys

from mpi4py import MPI
from pathlib import Path

from SimulationClass import GravitySimulation

# argv: Np, sim_time, output_level, output_dir, seed, debug
if __name__ == "__main__":
    # ------------------------------------------------------------
    # Check Inputs

    if len(sys.argv) < 5:
        print("Usage: python main.py NumParticles SimulationTime VerboseLvl OutputDir [Seed] [Debug]")
        exit(1)

    if not sys.argv[1].isnumeric():
        print("Argument 1 (Number of particles) must be an integer.")
        exit(1)
    Np = int(sys.argv[1])

    if not sys.argv[2].isnumeric():
        print("Argument 2 (Simulation time) must be an integer.")
        exit(1)
    sim_time = sys.argv[2] # Not cast to integer just yet (important).

    if not sys.argv[3].isnumeric():
        print("Argument 3 (Verbose level) must be an integer.")
        exit(1)
    verbose_level = int(sys.argv[3])

    output_dir = os.path.join(os.getcwd(), os.path.normpath(sys.argv[4]))

    s = None
    if len(sys.argv) >= 6: 
        if not sys.argv[5].isnumeric():
            print("Argument 5 (RNG Seed) must be an integer.")
        s = int(sys.argv[5])

    debug_flag = 0
    if len(sys.argv) >= 7:
        if not sys.argv[6].isnumeric():
            print("Argument 6 (Debug flag) must be an integer.")
        debug_flag = int(sys.argv[6])

    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # Begin Simulation

    sim = GravitySimulation(Mp, RS, xc, Np, L, Nc, s)

    while sim_time.isnumeric():
        sim.evolve_system(int(sim_time), output_dir, verbose_level, debug_flag)
        sim_time = input("Enter another time to continue simulation, or enter any key to quit")


