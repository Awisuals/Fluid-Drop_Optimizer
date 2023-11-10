# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 09:06:44 2023

@author: anter
"""
"""
Lets start over with a clean table. Define function 
(F for free surface energy) F = F(h_bar), where
h_bar is the vector for the height of grid points.

More simply:
    F = F({h_ij}) = f(h_00, h_01, ..., h_lxly)
"""
import numpy as np
from numpy.linalg import norm
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping
from scipy.optimize import minimize_scalar
from plotting import *
from functions import *
import time

# Import point clouds.
# Point cloud represents water droplet on a table.
# Points generated in Ellipsoid_Mesh_Kart - script.
# 2-D array with x,y,z coordinates for each point
# With different grid sizes.

def V_calc_test():
    
    volyme=[]
    free_energy=[]
    
    i=10
    while i<101:
        SIZE=i
        h0_1 = np.load(f'meshgrid_h0{SIZE}.npy')
        # h0_1_opt = Optimization(h0_1)
        # test1(h0_1_opt)
        
        # Test Volyme calculation
        # Import
        GRID=np.sqrt(len(h0_1))
        print(f"Grid size: {GRID} x {GRID}")
        print("System volyme:      " + str(System_Volyme(h0_1)))
        
        # Test Free energy calculation
        # F_calc_test(h0_data)
        print("System free energy: " + str(System_Free_Energy(h0_1)))
        
        volyme.append(System_Volyme(h0_1))
        free_energy.append(System_Free_Energy(h0_1))
        i+=10
    
    
    xes = np.linspace(10, i, num=int(np.sqrt(i)))
    plt.scatter(xes, free_energy)
    plt.scatter(xes, volyme)
    plt.title("Free energy and volyme as a function of grid size")
    plt.show()


def F_calc_test(h0_data):
    
    # Variable where to save system free energies 
    # at different grid sizes
    free_energy=np.empty(0)
    
    for data_id, data in enumerate(h0_data):
        data_faces = Triangulate_Data(data)
        print("Grid number: " + str(data_id) + "\n" + "Free energy: " + str(System_Free_Energy(data, data_faces)))
        free_energy = np.append(free_energy, System_Free_Energy(data, data_faces))
    # Remenber to change according to max grid size
    xes = np.linspace(10, 200, num=20)
    
    plt.scatter(xes, free_energy)
    plt.title("Free energy as a function of grid size w/o normalization")
    plt.show()
 

class MyStepFunction:
    
    def __init__(self, stepsize=0.5):
       self.stepsize = stepsize
       self.rng = np.random.default_rng()

    def __call__(self, x):
        s = self.stepsize
        
        for i in x:
            i += self.rng.uniform(0, 2*abs(i)*s) 
       
        return x


def Optimization_basinhopping(h0, GRID_SIZE, volume_constrt, N):
    
    contact_line = Morf_Contactline(h0, GRID_SIZE, N)
    h0_boundary = Compose_h0(h0, GRID_SIZE)
    boundary_uniq = Find_Grid_Boundary(h0_boundary)
    h0z_over0_indices = find_indices_greater_than_zero(h0)
    bounds=[(0,1)]
    cons = ({'type': 'eq', 'fun': lambda h0: System_Volyme(h0, GRID_SIZE)-volume_constrt},
            {'type': 'eq', 'fun': lambda h0: h0[boundary_uniq]},
            {'type': 'ineq', 'fun': lambda h0: h0[h0z_over0_indices] - 0.0002},
            {'type': 'eq', 'fun': lambda h0: h0[contact_line]})
    minimizer_kwargs = {"method":"trust-constr", # SLSQP trust-constr 
                        "constraints":cons, 
                        "args":GRID_SIZE, 
                        "bounds":bounds
                        } # "hess":True , "jac":True
    h0_opt = basinhopping(System_Free_Energy, h0, 
                       minimizer_kwargs=minimizer_kwargs, 
                       niter=200, 
                       T=0.1,
                       stepsize=0.01,
                       niter_success=5,
                       interval=2,
                       disp=True,
                       target_accept_rate=0.2,
                       stepwise_factor=0.9,
                       take_step = MyStepFunction())
    return h0_opt


def Optimize(N, M=0):
    
    GRID_SIZE=N
    h0_1z = np.load(f'FD_NO-REST_meshgrid_h0{GRID_SIZE}.npy')[:,2]
    h0_2z = np.load(f'FD_REST_meshgrid_h0{GRID_SIZE}.npy')[:,2]
    print(f"Grid size: {int(GRID_SIZE)} x {int(GRID_SIZE)}")
    h0_1 = Compose_h0(h0_1z, GRID_SIZE)
    h0_2 = Compose_h0(h0_2z, GRID_SIZE)
    
    Rest_System_volyme = System_Volyme(h0_2z, GRID_SIZE)
    Rest_System_free_energy = System_Free_Energy(h0_2z, GRID_SIZE)
    
    NonRest_System_volyme = System_Volyme(h0_1z, GRID_SIZE)
    NonRest_System_free_energy = System_Free_Energy(h0_1z, GRID_SIZE)
    
    # Test Volyme calculation
    print("Rest System volyme:      " + str(Rest_System_volyme))
    # Test Free energy calculation
    print("Rest System free energy: " + str(Rest_System_free_energy))
    
    # Test Volyme calculation
    print("Non-Rest System volyme:      " + str(NonRest_System_volyme))
    # Test Free energy calculation
    print("Non-Rest System free energy: " + str(NonRest_System_free_energy))
    
    start = time.time()
    h0_opt=Optimization_basinhopping(h0_1z, GRID_SIZE, NonRest_System_volyme, M)
    h0_opt_z=h0_opt.x
    end = time.time()
    runtime = end - start
    print(f"Elapsed optimization time: {runtime}")
    
    print("Optimized System volyme:      " + str(System_Volyme(h0_opt_z, GRID_SIZE)))
    print("Optimized System free energy: " + str(System_Free_Energy(h0_opt_z, GRID_SIZE)))
    
    h0_opt_plot = Compose_h0(h0_opt_z, GRID_SIZE)
    
    # test0(h0_opt_plot)
    test1(h0_opt_plot)
    
    test1(h0_1) # given h0
    test1(h0_2) # rest form of h0
    
    return h0_opt_plot, runtime

Optimize(10, 2)

def Morf_Contactline(h0, gs, N):
    
    nonzero_indices = np.nonzero(h0)[0]
    add_index = nonzero_indices[0]
    contact_line = [add_index]
    
    i=0
    while i<N:
        add_index += gs
        contact_line.append(add_index)
        i += 1
    return contact_line

# Grid_Sizes = [10,11,12,13,14,15,16,17,18,19,20]# ,21,22,23,24,25,26,27,28,29,30]
# Elapsed_Time = []
# Elapsed_Time_Grid = []

# for size in Grid_Sizes:
#     h0_opt_plot, runtime = Optimize(size)
#     Elapsed_Time.append(runtime)
#     Elapsed_Time_Grid.append(size)


#%%

GRID_SIZE=100
h0_1z = np.load(f'FD_NO-REST_meshgrid_h0{GRID_SIZE}.npy')[:,2]
h0_2z = np.load(f'FD_REST_meshgrid_h0{GRID_SIZE}.npy')[:,2]
print(f"Grid size: {int(GRID_SIZE)} x {int(GRID_SIZE)}")
h0_1 = Compose_h0(h0_1z, GRID_SIZE)
h0_2 = Compose_h0(h0_2z, GRID_SIZE)

Morf_Contactline(h0_2z, GRID_SIZE, 2)


#%%

fig = plt.figure()
ax1 = fig.add_subplot(211)
plt.plot(Elapsed_Time_Grid, Elapsed_Time)
plt.show()

