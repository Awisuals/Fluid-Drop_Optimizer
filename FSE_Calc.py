# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 09:06:44 2023

@author: anter

This script provides functionality to:
1) Load point-cloud data representing a droplet (h0).
2) Calculate droplet volume and free energy for varying grid sizes.
3) Perform a basinhopping global optimization to minimize free energy 
   subject to volume and boundary constraints.
4) Visualize and compare results.
"""

import numpy as np
from numpy.linalg import norm
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping, minimize_scalar
import time

from plotting import *       # Custom plotting module
from functions import *      # Custom functions module

# Named constant for minimum allowed droplet height in constraints
MIN_HEIGHT = 0.0002

def v_calc_test():
    """
    Test and visualize how volume and free energy vary with grid size.

    This function:
    1) Iterates over a sequence of grid sizes (10 to 100).
    2) Loads mesh data from .npy files.
    3) Prints the volume and free energy for each loaded shape.
    4) Stores results in lists and plots them as a function of grid size.
    """

    volumes = []
    free_energies = []

    # Loop over grid sizes from 10 to 100 in increments of 10
    for size in range(10, 101, 10):
        h0_data = np.load(f'meshgrid_h0{size}.npy')

        # Calculate the grid dimension from total points
        grid_dim = int(np.sqrt(len(h0_data)))
        print(f"Grid size: {grid_dim} x {grid_dim}")

        # Volume
        vol = System_Volyme(h0_data)
        print("System volume:      " + str(vol))

        # Free Energy
        f_energy = System_Free_Energy(h0_data)
        print("System free energy: " + str(f_energy))

        volumes.append(vol)
        free_energies.append(f_energy)

    # Create x-values to match the loop range
    # (the last loop value is 100, so the final size is 110 in code range(10, 101, 10)? 
    # Actually, range(10,101,10) stops at 100. We'll just reuse the same logic.)
    x_vals = list(range(10, 101, 10))

    plt.scatter(x_vals, free_energies, label='Free Energy')
    plt.scatter(x_vals, volumes, label='Volume')
    plt.title("Free Energy and Volume as a function of grid size")
    plt.legend()
    plt.show()


def f_calc_test(h0_data):
    """
    Plot free energy for a series of data arrays (h0_data).

    Parameters
    ----------
    h0_data : list or array-like
        A collection of droplet data arrays.
    """

    free_energies = np.empty(0)

    for data_id, data in enumerate(h0_data):
        data_faces = Triangulate_Data(data)
        f_energy = System_Free_Energy(data, data_faces)
        print("Grid number: " + str(data_id))
        print("Free energy: " + str(f_energy))
        free_energies = np.append(free_energies, f_energy)

    # For demonstration, we assume we have 20 data sets (10 to 200 in steps of 10).
    xes = np.linspace(10, 200, num=20)

    plt.scatter(xes, free_energies)
    plt.title("Free Energy as a function of grid size w/o normalization")
    plt.show()


class MyStepFunction:
    """
    Custom step function for basinhopping, which randomly perturbs
    each element of the solution vector within a scaled range.
    """

    def __init__(self, stepsize=0.5):
        """
        Parameters
        ----------
        stepsize : float, optional
            Multiplier for random step amplitude.
        """
        self.stepsize = stepsize
        self.rng = np.random.default_rng()

    def __call__(self, x):
        """
        Randomly perturb each element of x in the range [0, 2*|x_i|*stepsize].

        Parameters
        ----------
        x : array_like
            Current solution vector.

        Returns
        -------
        x : array_like
            Modified solution vector.
        """
        s = self.stepsize
        # IMPORTANT: We need to directly modify x's elements, not 'i' alone.
        for idx in range(len(x)):
            x[idx] += self.rng.uniform(0, 2 * abs(x[idx]) * s)
        return x


def optimization_basinhopping(h0, grid_size, volume_constraint, num_contact_pts):
    """
    Perform a basinhopping global optimization to minimize free energy of a droplet.

    Parameters
    ----------
    h0 : array_like
        Initial droplet shape data (flattened).
    grid_size : int
        The dimension of the grid (e.g., 10x10).
    volume_constraint : float
        Desired volume to maintain.
    num_contact_pts : int
        Number of points to enforce as part of the contact line.

    Returns
    -------
    h0_opt : OptimizeResult
        The result object from basinhopping, containing optimized shape, etc.
    """

    # Identify boundary/contact line constraints
    contact_line = morph_contact_line(h0, grid_size, num_contact_pts)
    h0_boundary = Compose_h0(h0, grid_size)
    boundary_uniq = Find_Grid_Boundary(h0_boundary)
    h0z_over0_indices = find_indices_greater_than_zero(h0)

    # Bound all h0 in [0, 1] for now
    bounds = [(0, 1)]

    # Constraints:
    # 1. The volume of droplet must match 'volume_constraint'.
    # 2. The boundary points must remain zero (contact line).
    # 3. All nonzero h0 elements >= MIN_HEIGHT.
    constraints = (
        {'type': 'eq', 'fun': lambda x: System_Volyme(x, grid_size) - volume_constraint},
        {'type': 'eq', 'fun': lambda x: x[boundary_uniq]},
        {'type': 'ineq', 'fun': lambda x: x[h0z_over0_indices] - MIN_HEIGHT},
    )

    minimizer_kwargs = {
        "method": "trust-constr",  # or SLSQP
        "constraints": constraints,
        "args": grid_size,
        "bounds": bounds
    }

    h0_opt = basinhopping(
        System_Free_Energy, 
        h0,
        minimizer_kwargs=minimizer_kwargs,
        niter=200,
        T=0.1,
        stepsize=0.01,
        niter_success=5,
        interval=2,
        disp=True,
        target_accept_rate=0.2,
        stepwise_factor=0.9,
        take_step=MyStepFunction()
    )

    return h0_opt


def optimize(grid_size_val, extra_param=0):
    """
    High-level function to:
    1. Load initial droplet shapes (REST and NO-REST).
    2. Compute volume/free energy of both.
    3. Run basinhopping optimization on the NO-REST shape
       with a volume constraint matching the NO-REST volume.
    4. Print and visualize results.

    Parameters
    ----------
    grid_size_val : int
        Grid dimension (e.g., 10 means 10x10).
    extra_param : int or float, optional
        Additional parameter that might serve as an extra constraint or 
        iteration count; usage depends on your needs.

    Returns
    -------
    h0_opt_plot : 2D array
        The optimized droplet shape in 2D form.
    runtime : float
        How long the optimization took (in seconds).
    """

    global GRID_SIZE
    GRID_SIZE = grid_size_val

    h0_no_rest_z = np.load(f'FD_NO-REST_meshgrid_h0{GRID_SIZE}.npy')[:, 2]
    h0_rest_z = np.load(f'FD_REST_meshgrid_h0{GRID_SIZE}.npy')[:, 2]

    print(f"Grid size: {GRID_SIZE} x {GRID_SIZE}")

    h0_no_rest_2d = Compose_h0(h0_no_rest_z, GRID_SIZE)
    h0_rest_2d = Compose_h0(h0_rest_z, GRID_SIZE)

    rest_system_volume = System_Volyme(h0_rest_z, GRID_SIZE)
    rest_system_free_energy = System_Free_Energy(h0_rest_z, GRID_SIZE)

    no_rest_system_volume = System_Volyme(h0_no_rest_z, GRID_SIZE)
    no_rest_system_free_energy = System_Free_Energy(h0_no_rest_z, GRID_SIZE)

    # Print volumes & free energies
    print("Rest System volume:      " + str(rest_system_volume))
    print("Rest System free energy: " + str(rest_system_free_energy))

    print("Non-Rest System volume:      " + str(no_rest_system_volume))
    print("Non-Rest System free energy: " + str(no_rest_system_free_energy))

    # Perform optimization
    start_time = time.time()
    h0_opt = optimization_basinhopping(h0_no_rest_z, GRID_SIZE, no_rest_system_volume, extra_param)
    h0_opt_z = h0_opt.x
    end_time = time.time()

    runtime = end_time - start_time
    print(f"Elapsed optimization time: {runtime}")

    print("Optimized System volume:      " + str(System_Volyme(h0_opt_z, GRID_SIZE)))
    print("Optimized System free energy: " + str(System_Free_Energy(h0_opt_z, GRID_SIZE)))

    # Convert optimized 1D array back to 2D for plotting
    h0_opt_plot = Compose_h0(h0_opt_z, GRID_SIZE)

    # Visualize results
    test1(h0_opt_plot)   # Plot optimized shape
    test1(h0_no_rest_2d) # Plot original NO-REST shape
    test1(h0_rest_2d)    # Plot REST shape

    return h0_opt_plot, runtime


def morph_contact_line(h0, grid_size_val, num_contact_pts):
    """
    Identify indices in h0 that define a 'contact line' region.

    Parameters
    ----------
    h0 : array_like
        Flattened droplet shape.
    grid_size_val : int
        Grid dimension (e.g., 10 means 10x10).
    num_contact_pts : int
        Number of points to force into the contact line.

    Returns
    -------
    contact_line_indices : list
        Indices in h0 that are designated as part of the contact line.
    """
    nonzero_indices = np.nonzero(h0)[0]
    add_index = nonzero_indices[0]
    contact_line_indices = [add_index]

    for _ in range(num_contact_pts):
        add_index += grid_size_val
        contact_line_indices.append(add_index)

    return contact_line_indices


# Example call
optimize(10, 0)

# Below are optional experiments, commented out:
# ----------------------------------------------------
# Grid_Sizes = [10,11,12,13,14,15,16,17,18,19,20]
# Elapsed_Time = []
# Elapsed_Time_Grid = []
#
# for size in Grid_Sizes:
#     h0_opt_plot, runtime = optimize(size)
#     Elapsed_Time.append(runtime)
#     Elapsed_Time_Grid.append(size)
#
# # Quick test for picking shape arrays
# GRID_SIZE = 10
# h0_1z = np.load(f'FD_NO-REST_meshgrid_h0{GRID_SIZE}.npy')[:,2]
# h0_2z = np.load(f'FD_REST_meshgrid_h0{GRID_SIZE}.npy')[:,2]
# print(f"Grid size: {int(GRID_SIZE)} x {int(GRID_SIZE)}")
# h0_1 = Compose_h0(h0_1z, GRID_SIZE)
# h0_2 = Compose_h0(h0_2z, GRID_SIZE)
# morph_contact_line(h0_2z, GRID_SIZE, 2)
#
# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# plt.plot(Elapsed_Time_Grid, Elapsed_Time)
# plt.show()
