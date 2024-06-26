# Python
import time

# MPL
import matplotlib        as mpl
import matplotlib.pyplot as plt

# NumPy
import numpy        as np
import numpy.random as random
import numpy.linalg as linalg
from numpy import cos, exp, log, pi, sin, sqrt

# FFTW 0.12.0
# There currently exists a bug with conda-forge's builds of FFTW 0.13.0+.
# Version 0.12.0 requires python version 3.10.
import pyfftw

from mpi4py import MPI

# This class defines everything that has to do with the discretized simulation space.
# In doing so these class methods perform the bulk of the work for this simulation.
class Grid:
    def __init__(self, L, Nc, Np, Mp):
        # Basic properties of the simulation space.
        # Self-explanatory. Nc is a number of cells per side length.
        self.L = L
        self.Nc = Nc
        self.dl = L/Nc
        self.Np = Np
        self.Mp = Mp

        # Array that stores the overlap between particles and grid cells.
        # CIC[0] and [1] store indices of lower and upper x boundaries (i1 and i2)
        # [2-3] and [4-5] store corresponding indices for y and z boundaries.
        # [6] and [7] store proportions of particle width contained in cells i1 and i2 (dx1 and dx2)
        # [8-9] and [10-11] store corresponding values for y and z.
        self.CIC_inds = np.empty((6,Np), dtype=int)
        self.CIC_props = np.empty((6,Np))

        # Memory-aligned arrays to store density field, potential field, and intermediate momentum representations.
        self.rhos = pyfftw.empty_aligned((Nc,Nc,Nc), dtype='float64')
        self.kspace = pyfftw.empty_aligned((Nc,Nc,Nc//2+1), dtype='complex128')
        self.phis = pyfftw.empty_aligned((Nc,Nc,Nc), dtype='float64')

        # Arrays to store derivatives of the potential field.
        self.fxs = np.empty((Nc,Nc,Nc))
        self.fys = np.empty((Nc,Nc,Nc))
        self.fzs = np.empty((Nc,Nc,Nc))

        # planA computes a forward DFT, used for transforming density field to k-space.
        # planB computes a backward DFT, used for transforming convolution to the potential field.
        self.planA = pyfftw.FFTW(self.rhos, self.kspace, axes=(0,1,2), direction='FFTW_FORWARD', flags=['FFTW_MEASURE','FFTW_DESTROY_INPUT'])
        self.planB = pyfftw.FFTW(self.kspace, self.phis, axes=(0,1,2), direction='FFTW_BACKWARD', flags=['FFTW_MEASURE','FFTW_DESTROY_INPUT'])

        G = 6.67e-11

        # Compute the gravitational kernel to implement the Poisson equation in k-space.
        def w(i,j,k): 
            return -4 * pi * G / ((2*sin(i*pi/Nc))**2 + (2*sin(j*pi/Nc))**2 + (2*sin(k*pi/Nc))**2)
        with np.errstate(divide='ignore'):
            self.wk = np.fromfunction(w, (Nc,Nc,Nc//2+1))
        self.wk[0,0,0] = 0

    # For internal use and testing use only.
    # Transforms real-space density field to complex momentum-space representation.
    def fft(self):
        return self.planA()

    # For internal use and testing use only.
    # Transforms momentum-space to real-space potential field.
    def ifft(self):
        return self.planB()

    # Clears the density field. This is the only array that necessarily needs to be written to zeros.
    # FFTW will hopefully always completely overwrite kspace and phis, and force assignment is extensive.
    def clear_grid(self):
        self.rhos.fill(0)

    # Uses current particle position data to assign particles to the CIC scheme.
    # Recall that the CIC scheme assumes that each particle is a cube with side length dl.
    def generate_CIC(self, array):
        x1 = array[0] - self.dl/2 # Left x boundary of the particle
        i1 = (x1//self.dl).astype(int) # Index of the cell containing the left boundary
        i2 = i1 + 1 # Index of the cell containing the right boundary
        xc = i2*self.dl # x coordinate of the border between left and right cells
        dx1 = (xc-x1)/self.dl # Percent of particle width contained in left cell
        dx2 = 1 - dx1 # Percent of particle width contained in right cell
        i2 = np.where(i2 >= self.Nc, 0, i2) # Enforce periodic boundary conditions by replacing Nc+1 with 0.
        # Negative indices should never be generated, and even if they are, they'll automatically index from the end.
        # Indices >Nc+1 should also never be generated mathematically, and the worst case numerical error just barely pushes it to Nc+2.
        # So it should do no harm to replace that with 0 as well.

        self.CIC_inds[0] = i1
        self.CIC_inds[1] = i2
        self.CIC_props[0] = dx1
        self.CIC_props[1] = dx2

        # print("x left boundary: ", x1)
        # print("x center boundary: ", xc)
        # print("x indices: ", i1, i2)
        # print("% in left: ", dx1)
        # print("% in right: ", dx2)

        # Repeat for y axis
        y1 = array[1] - self.dl/2
        j1 = (y1//self.dl).astype(int)
        j2 = j1 + 1
        yc = j2*self.dl
        dy1 = (yc-y1)/self.dl
        dy2 = 1 - dy1
        j2 = np.where(j2 >= self.Nc, 0, j2)

        self.CIC_inds[2] = j1
        self.CIC_inds[3] = j2
        self.CIC_props[2] = dy1
        self.CIC_props[3] = dy2

        # print("y bottom boundary: ", y1)
        # print("y center boundary: ", yc)
        # print("y indices: ", j1, j2)
        # print("% in bottom: ", dy1)
        # print("% in top: ", dy2)

        # Repeat for z axis
        z1 = array[2] - self.dl/2
        k1 = (z1//self.dl).astype(int)
        k2 = k1 + 1
        zc = k2*self.dl
        dz1 = (zc-z1)/self.dl
        dz2 = 1 - dz1
        k2 = np.where(k2 >= self.Nc, 0, k2)

        self.CIC_inds[4] = k1
        self.CIC_inds[5] = k2
        self.CIC_props[4] = dz1
        self.CIC_props[5] = dz2

        # print("z bottom boundary: ", z1)
        # print("z center boundary: ", zc)
        # print("z indices: ", k1, k2)
        # print("% in left: ", dz1)
        # print("% in right: ", dz2)

    # This uses the CIC scheme to distribute mass densities across the grid.
    def assign_positions(self):
        self.clear_grid()

        i1 = self.CIC_inds[0]
        i2 = self.CIC_inds[1]
        j1 = self.CIC_inds[2]
        j2 = self.CIC_inds[3]
        k1 = self.CIC_inds[4]
        k2 = self.CIC_inds[5]
        dx1 = self.CIC_props[0]
        dx2 = self.CIC_props[1]
        dy1 = self.CIC_props[2]
        dy2 = self.CIC_props[3]
        dz1 = self.CIC_props[4]
        dz2 = self.CIC_props[5]

        np.add.at(self.rhos,(i1,j1,k1),dx1*dy1*dz1)
        np.add.at(self.rhos,(i1,j2,k1),dx1*dy2*dz1)
        np.add.at(self.rhos,(i2,j1,k1),dx2*dy1*dz1)
        np.add.at(self.rhos,(i2,j2,k1),dx2*dy2*dz1)
        np.add.at(self.rhos,(i1,j1,k2),dx1*dy1*dz2)
        np.add.at(self.rhos,(i1,j2,k2),dx1*dy2*dz2)
        np.add.at(self.rhos,(i2,j1,k2),dx2*dy1*dz2)
        np.add.at(self.rhos,(i2,j2,k2),dx2*dy2*dz2)

        # This updates the array with percentages of particle in each cell.
        # So it needs to be multiplied by per particle density.
        self.rhos *= self.Mp
        self.rhos /= (self.dl)**3

        # print("i=0, j=0, k=0: ", rhos[0][0][0])
        # print("i=0, j=1, k=0: ", rhos[0][1][0])
        # print("i=1, j=0, k=0: ", rhos[1][0][0])
        # print("i=1, j=1, k=0: ", rhos[1][1][0])
        # print("i=0, j=0, k=1: ", rhos[0][0][1])
        # print("i=0, j=1, k=1: ", rhos[0][1][1])
        # print("i=1, j=0, k=1: ", rhos[1][0][1])
        # print("i=1, j=1, k=1: ", rhos[1][1][1])

    # Create a momentum-space representation of the density field.
    # Convolve with the grav kernel.
    # Transform back to real-space as the potential field.
    def solve_potential(self):
        rhoks = self.fft()
        rhoks *= self.wk
        self.ifft()
        # Why isn't the normalization factor Nc^3? idk.
        self.phis /= self.Nc**2

    # Finite difference the potential field.
    def solve_forces(self):
        # Finite difference along x, with periodic BCs.
        self.fxs[1:-1,:,:] = self.phis[:-2,:,:] - self.phis[2:,:,:]
        self.fxs[0,:,:] = self.phis[-1,:,:] - self.phis[1,:,:]
        self.fxs[-1,:,:] = self.phis[-2,:,:] - self.phis[0,:,:]
        self.fxs /= (2*self.dl)

        # Finite difference along y, with periodic BCs.
        self.fys[:,1:-1,:] = self.phis[:,:-2,:] - self.phis[:,2:,:]
        self.fys[:,0,:] = self.phis[:,-1,:] - self.phis[:,1,:]
        self.fys[:,-1,:] = self.phis[:,-2,:] - self.phis[:,0,:]
        self.fys /= (2*self.dl)

        # Finite difference along z, with periodic BCs.
        self.fzs[:,:,1:-1] = self.phis[:,:,:-2] - self.phis[:,:,2:]
        self.fzs[:,:,0] = self.phis[:,:,-1] - self.phis[:,:,1]
        self.fzs[:,:,-1] = self.phis[:,:,-2] - self.phis[:,:,0]
        self.fzs /= (2*self.dl)

    # This uses the CIC scheme to distribute grid force values to the particles array.
    def assign_accelerations(self, array):
        i1 = self.CIC_inds[0]
        i2 = self.CIC_inds[1]
        j1 = self.CIC_inds[2]
        j2 = self.CIC_inds[3]
        k1 = self.CIC_inds[4]
        k2 = self.CIC_inds[5]
        dx1 = self.CIC_props[0]
        dx2 = self.CIC_props[1]
        dy1 = self.CIC_props[2]
        dy2 = self.CIC_props[3]
        dz1 = self.CIC_props[4]
        dz2 = self.CIC_props[5]

        # Assign f_x to a_x.
        array[6] = dx1*dy1*dz1*self.fxs[i1,j1,k1]
        array[6] += dx1*dy1*dz2*self.fxs[i1,j1,k2]
        array[6] += dx1*dy2*dz1*self.fxs[i1,j2,k1]
        array[6] += dx1*dy2*dz2*self.fxs[i1,j2,k2]
        array[6] += dx2*dy1*dz1*self.fxs[i2,j1,k1]
        array[6] += dx2*dy1*dz2*self.fxs[i2,j1,k2]
        array[6] += dx2*dy2*dz1*self.fxs[i2,j2,k1]
        array[6] += dx2*dy2*dz2*self.fxs[i2,j2,k2]

        # Assign f_y to a_y.
        array[7] = dx1*dy1*dz1*self.fys[i1,j1,k1]
        array[7] += dx1*dy1*dz2*self.fys[i1,j1,k2]
        array[7] += dx1*dy2*dz1*self.fys[i1,j2,k1]
        array[7] += dx1*dy2*dz2*self.fys[i1,j2,k2]
        array[7] += dx2*dy1*dz1*self.fys[i2,j1,k1]
        array[7] += dx2*dy1*dz2*self.fys[i2,j1,k2]
        array[7] += dx2*dy2*dz1*self.fys[i2,j2,k1]
        array[7] += dx2*dy2*dz2*self.fys[i2,j2,k2]

        # Assign f_z to a_z.
        array[8] = dx1*dy1*dz1*self.fzs[i1,j1,k1]
        array[8] += dx1*dy1*dz2*self.fzs[i1,j1,k2]
        array[8] += dx1*dy2*dz1*self.fzs[i1,j2,k1]
        array[8] += dx1*dy2*dz2*self.fzs[i1,j2,k2]
        array[8] += dx2*dy1*dz1*self.fzs[i2,j1,k1]
        array[8] += dx2*dy1*dz2*self.fzs[i2,j1,k2]
        array[8] += dx2*dy2*dz1*self.fzs[i2,j2,k1]
        array[8] += dx2*dy2*dz2*self.fzs[i2,j2,k2]

    # Main loop that does everything.
    # Assumed that this is called at the end of every logical time step.
    def update_accelerations(self, array):
        # First, use the new position distribution to generate a CIC scheme.
        self.generate_CIC(array)
        # Next, create a new density distribution from the CIC scheme.
        self.assign_positions()
        # Solve the Poisson equation in momentum-space to get the potential field.
        self.solve_potential()
        # Differentiate the potential field to get the force field.
        self.solve_forces()
        # Update accelerations for each particle from the force field, using the CIC scheme.
        self.assign_accelerations(array)

    # This demonstrates the spherical distribution as it maps to the grid.
    # Displays a y-z slice taken from roughly the center of the sphere.
    def display_density(self):
        plt.imshow(self.rhos[self.Nc//2].T, extent=[0,self.L,0,self.L], origin='lower', cmap='inferno', alpha=0.9)
        plt.colorbar(label='Density (kg/m^3)')
        plt.show()

    # This plots the force (per unit mass) field as a function of distance from the center of mass.
    def display_forces(self, array, fit=False):
        # Center of mass coordinates.
        cmx = np.average(array[0])
        cmy = np.average(array[1])
        cmz = np.average(array[2])

        # Get coordinates for each cell center, and calculate their distances from the center of mass.
        def celldist(i,j,k): 
            return sqrt((cmx - (i+1/2)*self.dl)**2 + (cmy - (j+1/2)*self.dl)**2 + (cmz - (k+1/2)*self.dl)**2)
        celldists = np.fromfunction(celldist, (self.Nc,self.Nc,self.Nc))

        # Approximation of the body's radius.
        Rmax = np.max(sqrt((array[0] - cmx)**2 + (array[1] - cmy)**2 + (array[2] - cmz)**2))
        # Replaced with dl if the body is smaller than the cell resolution. (Mostly relevant for single particle case.)
        Rmax = max(Rmax,self.dl)

        # Magnitude of the force field.
        cellforces = sqrt((self.fxs)**2 + (self.fys)**2 + (self.fzs)**2)

        # Plot |f| as a function of r.
        cellrs = np.ravel(celldists)
        cellfs = np.ravel(cellforces)

        # Sort these arrays by r.
        p = cellrs.argsort()
        scellrs = cellrs[p]
        scellfs = cellfs[p]

        # Get the index of cell resolution size.
        cri = np.argmax(scellrs > self.dl)

        plt.scatter(scellrs, scellfs, alpha=0.1)

        GM = 6.67e-11*self.Mp*self.Np

        if fit:
            # Form of the theoretical force field of a solid spherical body. Physical for a = GM.
            def fit_func(r,a):
                with np.errstate(divide='ignore'):
                    return np.where(r < Rmax, a*r / Rmax**3, a / r**2)

            # Fitting the data to the theoretical force field. Start searching at a = GM.
            # Disregard all data for r smaller than the grid resolution size.
            popt, pcov = curve_fit(fit_func, scellrs[cri:], scellfs[cri:], p0=GM)

            # Plot the theory curve and fit curve.
            plt.plot(scellrs[cri:], fit_func(scellrs[cri:], GM), label="Theoretical", color="g")
            plt.plot(scellrs[cri:], fit_func(scellrs[cri:], popt[0]), label="Fitted", color="r")

        plt.xlabel('Distance from C.o.M. (Meters)')
        plt.ylabel('Gravity force per unit mass (N/kg)')

        # x Limit: 4 times body size.
        # y Limit: Force at 1.1 times body size.
        plt.xlim(0,4*Rmax)
        plt.ylim(0,1.1*fit_func(Rmax, GM))

        plt.axvline(Rmax, label="Surface", linestyle=":", alpha=0.2)

        plt.legend()
        plt.show()

        if fit:
            print(f"F_fit: {popt[0]:.3}\n Expected: {GM:.3}")

class GravitySimulation:
    # Mp: Mass of each particle. (kg)
    # Np: Total number of particles in the simulation.
    # L: Side length of the (cubical) simulation space.(m)
    # Nc: Number of cells on each dimension of the simulation space lattice. (m)
    def __init__(self, parts, Np, Mp, L, Nc):
        # Set up the simulation space.
        self.Mp = Mp
        self.grid = Grid(L,Nc,Np,Mp)
        self.parts = parts
        
        self.grid.update_accelerations(self.parts)

        self.dT = 2 # Placeholder value
        self.T = 0 # Current simulation time.

    def initial_step(self):
        # First kick
        self.parts[3] = self.parts[6] * self.dT/2
        self.parts[4] = self.parts[7] * self.dT/2
        self.parts[5] = self.parts[8] * self.dT/2

        # First drift
        self.parts[0] += self.parts[3] * self.dT
        self.parts[1] += self.parts[4] * self.dT
        self.parts[2] += self.parts[5] * self.dT

        # Update acceleration
        self.grid.update_accelerations(self.parts)

    def time_step(self):
        # 2 kicks
        self.parts[3] += self.parts[6] * self.dT
        self.parts[4] += self.parts[7] * self.dT
        self.parts[5] += self.parts[8] * self.dT

        # Drift
        self.parts[0] += self.parts[3] * self.dT
        self.parts[1] += self.parts[4] * self.dT
        self.parts[2] += self.parts[5] * self.dT

        # Update acceleration
        self.grid.update_accelerations(self.parts)

    def final_step(self, debug):
        # One final kick.
        self.parts[3] += self.parts[6] * self.dT/2
        self.parts[4] += self.parts[7] * self.dT/2
        self.parts[5] += self.parts[8] * self.dT/2

        if debug:
            print(f"Max x vel: {np.max(self.parts[3])}")
            print(f"Max x acc: {np.max(self.parts[6])}")
            print(f"Recommended time step: {min( (1/128)/ np.max(self.parts[3]) , sqrt( (1/128) / np.max(self.parts[6]) ) )}")


    # sim_time: Duration of the simulation, in seconds.
    # verbose: 1 = output position visualization, 2 = output other visualizations.
    # debug: 1 = print time step info to console.
    # Future work - allow reverse evolution. Would that be useful for testing stability?
    def evolve_system(self, sim_time, output_dir, verbose=1, debug=0):
        timesteps = int(sim_time // self.dT)
        snapshot_steps = 10

        avg_compute_time = 0
        time1 = 0
        time2 = 0

        if debug:
            time1 = time.time()

        for _ in range(timesteps // snapshot_steps):

            self.initial_step()

            for _ in range(2, snapshot_steps+1):
                self.time_step()

            self.final_step(debug)

            self.T += snapshot_steps*self.dT 

            if verbose:
                if debug:
                    time2 = time.time()
                    avg_compute_time += time2 - time1
                self.output_positions(output_dir)
                if debug:
                    time1 = time.time()

        self.initial_step()

        for _ in range(2, (timesteps % snapshot_steps) + 1):
            self.time_step()

        self.final_step(debug)

        self.T += (timesteps % snapshot_steps)*self.dT

        if debug:
            time2 = time.time()
            avg_compute_time += time2 - time1
            avg_compute_time /= timesteps
            avg_compute_time *= self.dT

        if verbose:
            self.output_positions(output_dir)

        print(f"Advanced simulation by {timesteps} time steps. Current time: {self.T} seconds.")
        if debug:
            print(f"Average computation time per simulation second: {avg_compute_time} seconds.")


    # A simple plot of the particle distribution in space.
    # Not quite to scale and doesn't support moving through the frame.
    # But well-suited for visualization within the notebook.
    def display_positions_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(self.parts[0], self.parts[1], self.parts[2])

        ax.set_xlabel('X (Meters)')
        ax.set_ylabel('Y (Meters)')
        ax.set_zlabel('Z (Meters)')

        ax.set_xlim(0,self.grid.L)
        ax.set_ylim(0,self.grid.L)
        ax.set_zlim(0,self.grid.L)

        plt.show()


    # This plots the force (per unit mass) field as a function of distance from the center of mass.
    def display_forces(self, fit=False):
        self.grid.display_forces(self.parts, fit)

    def output_positions(self, output_dir):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
        axes[0].scatter(self.parts[0], self.parts[1])
        axes[0].set_xlim(0,self.grid.L)
        axes[0].set_ylim(0,self.grid.L)
        axes[0].set_xlabel("x Position (m)")
        axes[0].set_ylabel("y Position (m)")
        axes[0].set_aspect('equal','box')
        axes[1].scatter(self.parts[1], self.parts[2])
        axes[1].set_xlim(0,self.grid.L)
        axes[1].set_ylim(0,self.grid.L)
        axes[1].set_xlabel("y Position (m)")
        axes[1].set_ylabel("z Position (m)")
        axes[1].set_aspect('equal','box')

        plt.suptitle(f'x vs y and y vs z at time T = {self.T} seconds.')
        plt.tight_layout()

        out = f"output_{self.T:03d}.pdf"

        out_path = output_dir.joinpath(out)

        plt.savefig(out_path,format="pdf",dpi=300)
        plt.close()


