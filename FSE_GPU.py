# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 13:37:27 2023

@author: anter

Write system free energy calculation GPU-acceleration in mind.
"""
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import basinhopping
from plotting import *
import time
import os
from numba import cuda
cuda.close()

# Import point clouds.
# Point cloud represents water droplet on a table.
# Points generated in Ellipsoid_Mesh_Kart - script.
# 2-D array with x,y,z coordinates for each point
# With different grid sizes.


"""
Define Here some utility functions not necessary to GPU accelerate.

"""

# class MyStepFunction:
    
#     def __init__(self, stepsize=0.5):
#         self.stepsize = stepsize
#         self.rng = np.random.default_rng()

#     def __call__(self, x):
#         s = self.stepsize
        
#         for i in x:
#             i += self.rng.uniform(0, 2*abs(i)*s) 
       
#         return x

class MyStepFunction:
    
    def __init__(self, stepsize=0.5):
        self.stepsize = stepsize
        self.rng = np.random.default_rng()

    def __call__(self, x):
        s = self.stepsize
        
        for i in range(len(x)):
            perturbation = self.rng.uniform(0, 2 * abs(x[i]) * s)
            x[i] += perturbation
       
        return x


def Find_Grid_Boundary(grid_data):
    """
    Finds the boundary coordinates of a grid.

    Parameters
    ----------
    grid_data : Array
        Some x, y - grid.

    Returns
    -------
    boundary_unique_values : Array
        Boundary coordinate indices of 
        grid_data.

    """
    
    # Determine the number of unique y-values
    unique_y_values = np.unique(grid_data[:, 1])
    
    # Determine the number of unique x-values
    unique_x_values = np.unique(grid_data[:, 0])
    
    # Assuming the grid is square, calculate the number of rows and columns
    num_rows = len(unique_y_values)
    num_columns = len(unique_x_values)
    
    # Determine the indices of the corners
    top_left_corner = 0
    top_right_corner = num_columns - 1
    bottom_left_corner = (num_rows - 1) * num_columns
    bottom_right_corner = num_rows * num_columns - 1
    
    # Determine the indices of the sides
    top_side_indices = np.arange(top_left_corner, top_right_corner + 1)
    bottom_side_indices = np.arange(bottom_left_corner, bottom_right_corner + 1) # , num_columns
    left_side_indices = np.arange(top_left_corner, bottom_left_corner + 1, num_columns)
    right_side_indices = np.arange(top_right_corner, bottom_right_corner + 1, num_columns)

    # Assuming you have calculated the side indices as before
    all_side_indices = np.concatenate([
        top_side_indices,
        bottom_side_indices,
        left_side_indices,
        right_side_indices])

    boundary_unique_values = np.unique(all_side_indices)
    return boundary_unique_values


def Triangulate_Data(data):
    """
    Triangulates given 3d point cloud and 
    returns the 2d simplices (triangles) that make up  the point cloud.

    Parameters
    ----------
    data : Array
        3d point cloud to be triangulated.

    Returns
    -------
    data_simplices : Array
        Triangulated simplices.

    """
    data_tri = Delaunay(data[:,:2])
    data_indices = data_tri.simplices
    data_simplices = data[data_indices]
    return data_simplices


def Compose_Model_XYZ_Points(h0z, GRID_SIZE):
    """
    Composes point cloud array for system model in
    x, y and z-coordinates.

    Parameters
    ----------
    h0z : Array
        z-values of model.
    GRID_SIZE : Array
        Matching grid size for z-values.

    Returns
    -------
    h0 : array
        Model point cloud x, y and z-coordinates.

    """
    h0xy = np.load(f'FD_REST_meshgrid_h0{GRID_SIZE}.npy')[:,:2]
    tmp=[]
    tmp.append(h0xy[:,0])
    tmp.append(h0xy[:,1])
    tmp.append(h0z)
    h0 = np.asarray(tmp).T
    return h0


def Morf_Contactline(h0, gs, N):
    """
    Attach some contactline points to zero 
    to see how it changes the shape of the droplet.
    Attached points are from the top of the model 
    (in x,y plane) straight in.

    Parameters
    ----------
    h0 : Array
        Model points z-values.
    gs : int
        Grid size.
    N : int
        How many points in we want to attach to zero.

    Returns
    -------
    contact_line : Array
        Indices of the points we 
        want to attach to zero.

    """
    
    nonzero_indices = np.nonzero(h0)[0]
    add_index = nonzero_indices[0]
    contact_line = [add_index]
    
    i=0
    while i<N:
        add_index += gs
        contact_line.append(add_index)
        i += 1
    return contact_line


def find_indices_greater_than_zero(arr):
    """
    Simple function to find nonzdero elements of array.

    Parameters
    ----------
    arr : Array
        array we want to find nonzero indices from.

    Returns
    -------
    Array
        nonzero elements of arr.

    """
    return np.where(arr > 0)[0]


@cuda.jit(device=True)
def Simplex_GE(sp, xy):
    """
    Claculate the gravitational energy of one simplex beam.

    Parameters
    ----------
    sp : TYPE
        DESCRIPTION.
    xy : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    Rho_W = 997
    g = 9.81
    h_ave = (sp[0, 2] + sp[1, 2] + sp[2, 2])/3
    return (1/4)*Rho_W*g*xy[0]*xy[0]*h_ave**2


@cuda.jit(device=True)
def Simplex_Area(sp):
    """
    For calculating area of triangle (2d-simplex) 
    from three points (vectors)
    
    Example of triangle area calculation:
        triangle = np.array([[1,1,1],[2,2,2],[3,1,2]])
        print(Area(triangle[0], triangle[1], triangle[2]))


    Parameters
    ----------
    a : Array
        Points for point A in 2d-simplex.
    b : TYPE
        Points for point B in 2d-simplex.
    c : TYPE
        Points for point C in 2d-simplex.

    Returns
    -------
    Float
        Area of triangle.

    """
    # Define points of the simplex
    a,b,c = sp[0,:], sp[1,:], sp[2,:]
    
    # Vectors formed by subtracting one point from another
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]

    # Cross product components
    cross_product_i = uy * vz - uz * vy
    cross_product_j = uz * vx - ux * vz
    cross_product_k = ux * vy - uy * vx

    # Magnitude of the cross product (area of the parallelogram)
    area_parallelogram = (cross_product_i**2 + cross_product_j**2 + cross_product_k**2)**0.5

    # Area of the triangle (half of the parallelogram area)
    area_simplex = 0.5 * area_parallelogram

    return area_simplex


@cuda.jit(device=True)
def Simplex_ST(sp):
    """
    For picing the right surface tension. 
    If any z-value on any point of the simplex 
    is greater than 0, use liquid-gas interface 
    surface tension. Else use solid-gas interface
    surface tension.

    Parameters
    ----------
    sp : 2d-Array
        Coordinates of the simplex points.

    Returns
    -------
    Float
        Right surface tension.

    """
    Gamma_LG = 0.076
    Gamma_SG = Gamma_LG*np.cos(0.3875)
    if (sp[0, 2] or sp[1, 2] or sp[2, 2]) > 0.0000005:
        return Gamma_LG
    else:
        return Gamma_SG


@cuda.jit(device=True)
def Simplex_FE(sp, xy):
    """
    For calculating the free energy of one simplex beam.

    Parameters
    ----------
    sp : 2d-Array
        Simplex point coordinates.
    xy : Array
        For storing delta_x and delta_y.

    Returns
    -------
    Float
        Free energy of the simplex beam.

    """
    return Simplex_ST(sp)*Simplex_Area(sp)+Simplex_GE(sp, xy)


@cuda.jit(device=True)
def Simplex_Volume(sp, xy):
    """
    Simple function to calculate the volyme of a simplex.

    Parameters
    ----------
    sp : Array
        Simplices.
    xy : Array
        x, y array for grid reference height.

    Returns
    -------
    Float
        Volyme of one simplex.

    """
    h_ave = (sp[0, 2] + sp[1, 2] + sp[2, 2])/3
    return (xy[0]*xy[0]*h_ave)/2


@cuda.jit
def System_FE_Kernel(h0_simplices, Delta_XY, result):
    i = cuda.grid(1)
    if i < len(h0_simplices):
        result[i] = Simplex_FE(h0_simplices[i, :, :], Delta_XY)
    return


@cuda.jit
def System_V_Kernel(h0_simplices, Delta_XY, result):
    i = cuda.grid(1)
    if i < len(h0_simplices):
        result[i] = Simplex_Volume(h0_simplices[i, :, :], Delta_XY)
    return


def Parallel_Sys_FE(h0z, GRID_SIZE):
    
    # Define input parameters
    h0 = Compose_Model_XYZ_Points(h0z, GRID_SIZE)
    h0_simplices = Triangulate_Data(h0)
    Delta_XY = [abs(h0[0, 0] - h0[1, 0]), abs(h0[0, 0] - h0[1, 0])]
    # Delta_XY = abs(h0[0, 0] - h0[1, 0])
    
    # Find amount of simplices for copying params to device
    n = len(h0_simplices)
    
    # Initialize the output array on the host
    result_host = np.zeros(n, dtype=np.float64)
    
    # Transfer input parameters to the device (GPU)
    h0_simplices_device = cuda.to_device(h0_simplices)
    Delta_XY_device = cuda.to_device(Delta_XY)
    
    # Initialize the output array on the device
    result_device = cuda.device_array(n, dtype=np.float64)
    
    # # Define the number of threads per block and blocks per grid
    threads_per_block = 1024 # 1024
    blocks_per_grid = 256 # (n + threads_per_block - 1) // threads_per_block
    
    # Launch the CUDA kernel
    System_FE_Kernel[blocks_per_grid, threads_per_block](
        h0_simplices_device, Delta_XY_device, result_device)
    
    # Transfer the result back to the host
    result_device.copy_to_host(result_host)
    
    # Return model free energy
    return result_host.sum()


def Parallel_Sys_V(h0z, GRID_SIZE):
    
    # Define input parameters
    h0 = Compose_Model_XYZ_Points(h0z, GRID_SIZE)
    h0_simplices = Triangulate_Data(h0)
    Delta_XY = [abs(h0[0, 0] - h0[1, 0]), abs(h0[0, 0] - h0[1, 0])]
    # Delta_XY = abs(h0[0, 0] - h0[1, 0])
    
    # Find amount of simplices for copying params to device
    n = len(h0_simplices)
    
    # Initialize the output array on the host
    result_host = np.zeros(n, dtype=np.float64)
    
    # Transfer input parameters to the device (GPU)
    h0_simplices_device = cuda.to_device(h0_simplices)
    Delta_XY_device = cuda.to_device(Delta_XY)
    
    # Initialize the output array on the device
    result_device = cuda.device_array(n, dtype=np.float64)
    
    # # Define the number of threads per block and blocks per grid
    threads_per_block = 1024 # 1024
    blocks_per_grid = 256 # (n + threads_per_block - 1) // threads_per_block
    
    # Launch the CUDA kernel
    System_V_Kernel[blocks_per_grid, threads_per_block](
        h0_simplices_device, Delta_XY_device, result_device)
    
    # Transfer the result back to the host
    result_device.copy_to_host(result_host)
    
    # Return model volume
    return result_host.sum()


def Optimization_basinhopping(h0, GRID_SIZE, volume_constrt, N):
    """
    Function which calls basinhopping 
    optimizing algorithm.

    Parameters
    ----------
    h0 : Array
        z values of model.
    GRID_SIZE : int
        Size of the grid.
    volume_constrt : float
        Value for volyme constraint.
    N : int
        For morfing the contact line.

    Returns
    -------
    h0_opt : TYPE
        DESCRIPTION.

    """
    
    contact_line = Morf_Contactline(h0, GRID_SIZE, N)
    boundary_uniq = Find_Grid_Boundary(Compose_Model_XYZ_Points(h0, GRID_SIZE))
    h0z_over0_indices = find_indices_greater_than_zero(h0)
    bounds=[(0,1)]
    cons = ({'type': 'eq', 'fun': lambda h0: Parallel_Sys_V(h0, GRID_SIZE)-volume_constrt},
            {'type': 'eq', 'fun': lambda h0: h0[boundary_uniq]},
            {'type': 'ineq', 'fun': lambda h0: h0[h0z_over0_indices] - 0.0002}) # 0.0002 is min height
    # ,{'type': 'eq', 'fun': lambda h0: h0[contact_line]}
    minimizer_kwargs = {"method":"trust-constr", # SLSQP trust-constr 
                        "constraints":cons, 
                        "args":GRID_SIZE, 
                        "bounds":bounds
                        } # "hess":True , "jac":True
    h0_opt = basinhopping(Parallel_Sys_FE, 
                          h0, 
                          minimizer_kwargs=minimizer_kwargs, 
                          niter=200, 
                          T=0.1,
                          stepsize=0.01,
                          niter_success=5,
                          interval=2,
                          disp=True,
                          target_accept_rate=0.2,
                          stepwise_factor=0.9)
                        #   take_step = MyStepFunction())
    return h0_opt


def main(N, M=0):
    
    GRID_SIZE=N
    h0_1z = np.load(f'FD_NO-REST_meshgrid_h0{GRID_SIZE}.npy')[:,2]
    h0_2z = np.load(f'FD_REST_meshgrid_h0{GRID_SIZE}.npy')[:,2]
    print(f"Grid size: {int(GRID_SIZE)} x {int(GRID_SIZE)}")
    h0_1 = Compose_Model_XYZ_Points(h0_1z, GRID_SIZE)
    h0_2 = Compose_Model_XYZ_Points(h0_2z, GRID_SIZE)
    
    rest_system_volyme = Parallel_Sys_V(h0_2z, GRID_SIZE)
    rest_system_free_energy = Parallel_Sys_FE(h0_2z, GRID_SIZE)
    
    nonrest_system_volyme = Parallel_Sys_V(h0_1z, GRID_SIZE)
    nonrest_system_free_energy = Parallel_Sys_FE(h0_1z, GRID_SIZE)
    
    # Test Volyme calculation
    print("Rest System volyme:      " + str(rest_system_volyme))
    # Test Free energy calculation
    print("Rest System free energy: " + str(rest_system_free_energy))
    
    # Test Volyme calculation
    print("Non-Rest System volyme:      " + str(nonrest_system_volyme))
    # Test Free energy calculation
    print("Non-Rest System free energy: " + str(nonrest_system_free_energy))
    
    start = time.time()
    h0_opt=Optimization_basinhopping(h0_1z, 
                                     GRID_SIZE, 
                                     nonrest_system_volyme, 
                                     M)
    h0_opt_z=h0_opt.x
    end = time.time()
    runtime = end - start
    print(f"Elapsed optimization time: {runtime}")
    
    print("Optimized System volyme:      " + str(Parallel_Sys_V(h0_opt_z, GRID_SIZE)))
    print("Optimized System free energy: " + str(Parallel_Sys_FE(h0_opt_z, GRID_SIZE)))
    
    h0_opt_points = Compose_Model_XYZ_Points(h0_opt_z, GRID_SIZE)
    
    
    method = "trustconstr"
    h0_file_name = 'FD_OPT_meshgrid_h0'+str(GRID_SIZE)+str(f'-GPU-{method}')

    if os.path.isfile(h0_file_name+'.npy'):
        os.remove(h0_file_name+'.npy')

    np.save(h0_file_name, h0_opt_points)

    SAVEFILE = f'RUNFILE-GPU-{method}.txt'

    with open(SAVEFILE, "a") as f:
        f.write("-" * 40 + "\n")
        f.write(f"Grid: {GRID_SIZE}x{GRID_SIZE}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Elapsed Optimization time: {runtime}\n")
        f.write(f"\n")
        
        f.write(f"Non-Rest System volume:   {nonrest_system_volyme}\n" )
        f.write(f"Optimized System volume:   {Parallel_Sys_V(h0_opt_z, GRID_SIZE)}\n")
        f.write(f"Rest System volume:   {rest_system_volyme}\n")
        
        f.write(f"\n")
        
        f.write(f"Non-Rest System free energy:  {nonrest_system_free_energy}\n")
        f.write(f"Optimized System free energy:  {Parallel_Sys_FE(h0_opt_z, GRID_SIZE)}\n")
        f.write(f"Rest System free energy:  {rest_system_free_energy}\n")
        
        f.write("\n")

    
    # test0(h0_opt_plot)
    # test1(h0_opt_plot) # Surface plot
    
    # test1(h0_1) # surface plot of given h0
    # test1(h0_2) # surface plot of  rest form of h0
    
    # plot_trisurf_faces(data1=h0_1,
    #                data2=h0_opt_points,
    #                data3=h0_2,
    #                z_scale=[0,0.0015],
    #                view_param=[15,45,8]
    #             #    title="Triangulated surface representation of a sessile droplet in equilibrium"
    #                )

    
    return h0_opt_points, runtime


# Call optimizer
grid_sizes = [10,11,12,13,14,15,16,17,18,19,20,
              21,22,23,24,25,26,27,28,29,30]

for i in grid_sizes:
    main(i)



