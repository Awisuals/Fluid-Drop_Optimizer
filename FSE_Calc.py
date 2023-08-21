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


# Lisää tilaa reunoille
# tiheämpi grid
# Alkutilan tilavuusyhteensopivuus
# Muita minimointi metodeja
def Optimization_basinhopping(h0, GRID_SIZE):
    
    h0_boundary = Compose_h0(h0, GRID_SIZE)
    boundary_uniq = Find_Grid_Boundary(h0_boundary)
    h0z_over0_indices = find_indices_greater_than_zero(h0)
    
    bounds=[(0,1)]
    cons = ({'type': 'eq', 'fun': lambda h0: System_Volyme(h0, GRID_SIZE)-2.6829739468141043e-09},
            {'type': 'eq', 'fun': lambda h0: h0[boundary_uniq]},
            {'type': 'ineq', 'fun': lambda h0: h0[h0z_over0_indices] - 0.00015})
    minimizer_kwargs = {"method":"SLSQP", # SLSQP trust-constr 
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
                       take_step = MyStepFunction()
                       )
    return h0_opt


GRID_SIZE=20
h0_1z = np.load(f'FD_NO-REST_meshgrid_h0{GRID_SIZE}.npy')[:,2]
h0_2z = np.load(f'FD_REST_meshgrid_h0{GRID_SIZE}.npy')[:,2]
print(f"Grid size: {int(GRID_SIZE)} x {int(GRID_SIZE)}")
h0_1 = Compose_h0(h0_1z, GRID_SIZE)
h0_2 = Compose_h0(h0_2z, GRID_SIZE)

# Test Volyme calculation
print("Rest System volyme:      " + str(System_Volyme(h0_2z, GRID_SIZE)))
# Test Free energy calculation
print("Rest System free energy: " + str(System_Free_Energy(h0_2z, GRID_SIZE)))

# Test Volyme calculation
print("Non-Rest System volyme:      " + str(System_Volyme(h0_1z, GRID_SIZE)))
# Test Free energy calculation
print("Non-Rest System free energy: " + str(System_Free_Energy(h0_1z, GRID_SIZE)))

h0_opt=Optimization_basinhopping(h0_1z, GRID_SIZE)
h0_opt_z=h0_opt.x

print("Optimized System volyme:      " + str(System_Volyme(h0_opt_z, GRID_SIZE)))
print("Optimized System free energy: " + str(System_Free_Energy(h0_opt_z, GRID_SIZE)))

h0_opt_plot = Compose_h0(h0_opt_z, GRID_SIZE)

test0(h0_opt_plot)
test1(h0_opt_plot)

test1(h0_1)
test1(h0_2)

