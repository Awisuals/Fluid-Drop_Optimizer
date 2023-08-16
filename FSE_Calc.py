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
import matplotlib as mpl
from scipy import spatial as sp_spatial
import mpl_toolkits.mplot3d as a3
from scipy.optimize import basinhopping
from scipy.optimize import minimize_scalar


# Import point clouds.
# Point cloud represents water droplet on a table.
# Points generated in Ellipsoid_Mesh_Kart - script.
# 2-D array with x,y,z coordinates for each point
# With different grid sizes.
h0_data = [np.load('meshgrid_h010.npy')
,np.load('meshgrid_h020.npy')
,np.load('meshgrid_h030.npy')
,np.load('meshgrid_h040.npy')
,np.load('meshgrid_h050.npy')
,np.load('meshgrid_h060.npy')
,np.load('meshgrid_h070.npy')
,np.load('meshgrid_h080.npy')
,np.load('meshgrid_h090.npy')
,np.load('meshgrid_h0100.npy')
,np.load('meshgrid_h0110.npy')
,np.load('meshgrid_h0120.npy')
,np.load('meshgrid_h0130.npy')
,np.load('meshgrid_h0140.npy')
,np.load('meshgrid_h0150.npy')
,np.load('meshgrid_h0160.npy')
,np.load('meshgrid_h0170.npy')
,np.load('meshgrid_h0180.npy')
,np.load('meshgrid_h0190.npy')
,np.load('meshgrid_h0200.npy')]
# ,np.load('meshgrid_h0210.npy')
# ,np.load('meshgrid_h0220.npy')
# ,np.load('meshgrid_h0230.npy')
# ,np.load('meshgrid_h0240.npy')
# ,np.load('meshgrid_h0250.npy')
# ,np.load('meshgrid_h0260.npy')
# ,np.load('meshgrid_h0270.npy')
# ,np.load('meshgrid_h0280.npy')
# ,np.load('meshgrid_h0290.npy')
# ,np.load('meshgrid_h0300.npy')]
# ,np.load('meshgrid_h0310.npy')
# ,np.load('meshgrid_h0320.npy')
# ,np.load('meshgrid_h0330.npy')
# ,np.load('meshgrid_h0340.npy')
# ,np.load('meshgrid_h0350.npy')
# ,np.load('meshgrid_h0360.npy')
# ,np.load('meshgrid_h0370.npy')
# ,np.load('meshgrid_h0380.npy')
# ,np.load('meshgrid_h0390.npy')
# ,np.load('meshgrid_h0400.npy')
# ,np.load('meshgrid_h0410.npy')
# ,np.load('meshgrid_h0420.npy')
# ,np.load('meshgrid_h0430.npy')
# ,np.load('meshgrid_h0440.npy')
# ,np.load('meshgrid_h0450.npy')
# ,np.load('meshgrid_h0460.npy')
# ,np.load('meshgrid_h0470.npy')
# ,np.load('meshgrid_h0480.npy')
# ,np.load('meshgrid_h0490.npy')
# ,np.load('meshgrid_h0500.npy')]
# ,np.load('meshgrid_h0510.npy')
# ,np.load('meshgrid_h0520.npy')
# ,np.load('meshgrid_h0530.npy')
# ,np.load('meshgrid_h0540.npy')
# ,np.load('meshgrid_h0550.npy')
# ,np.load('meshgrid_h0560.npy')
# ,np.load('meshgrid_h0570.npy')
# ,np.load('meshgrid_h0580.npy')
# ,np.load('meshgrid_h0590.npy')
# ,np.load('meshgrid_h0600.npy')
# ,np.load('meshgrid_h0610.npy')
# ,np.load('meshgrid_h0620.npy')
# ,np.load('meshgrid_h0630.npy')
# ,np.load('meshgrid_h0640.npy')
# ,np.load('meshgrid_h0650.npy')
# ,np.load('meshgrid_h0660.npy')
# ,np.load('meshgrid_h0670.npy')
# ,np.load('meshgrid_h0680.npy')
# ,np.load('meshgrid_h0690.npy')
# ,np.load('meshgrid_h0700.npy')
# ,np.load('meshgrid_h0710.npy')
# ,np.load('meshgrid_h0720.npy')
# ,np.load('meshgrid_h0730.npy')
# ,np.load('meshgrid_h0740.npy')
# ,np.load('meshgrid_h0750.npy')
# ,np.load('meshgrid_h0760.npy')
# ,np.load('meshgrid_h0770.npy')
# ,np.load('meshgrid_h0780.npy')
# ,np.load('meshgrid_h0790.npy')
# ,np.load('meshgrid_h0800.npy')
# ,np.load('meshgrid_h0810.npy')
# ,np.load('meshgrid_h0820.npy')
# ,np.load('meshgrid_h0830.npy')
# ,np.load('meshgrid_h0840.npy')
# ,np.load('meshgrid_h0850.npy')
# ,np.load('meshgrid_h0860.npy')
# ,np.load('meshgrid_h0870.npy')
# ,np.load('meshgrid_h0880.npy')
# ,np.load('meshgrid_h0890.npy')
# ,np.load('meshgrid_h0900.npy')
# ,np.load('meshgrid_h0910.npy')
# ,np.load('meshgrid_h0920.npy')
# ,np.load('meshgrid_h0930.npy')
# ,np.load('meshgrid_h0940.npy')
# ,np.load('meshgrid_h0950.npy')
# ,np.load('meshgrid_h0960.npy')
# ,np.load('meshgrid_h0970.npy')
# ,np.load('meshgrid_h0980.npy')
# ,np.load('meshgrid_h0990.npy')
# ,np.load('meshgrid_h01000.npy')
# ]


# Some tplotting funktions
def test0(h0_1):
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(h0_1[:,0],
               h0_1[:,1],
               h0_1[:,2])
    ax.set_zlim([0, 0.0025])
    plt.show()


def test1(h0_1):
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_trisurf(h0_1[:,0],
                    h0_1[:,1],
                    h0_1[:,2],linewidth=0.2,antialiased=True)
    ax.set_zlim([0, 0.0025])
    plt.show()


# test1(h0_1)


def test2(h0_1_faces):
    ax1 = plt.figure().add_subplot(projection='3d')
    i=0
    while i<len(h0_1_faces):
        ax1.plot_trisurf(h0_1_faces[i,:,0],
                         h0_1_faces[i,:,1],
                         h0_1_faces[i,:,2],linewidth=0.2,antialiased=True)
        i+=1
    # ax1.set_zlim([0, 0.0025])
    plt.show()


def test3(h0_1_faces):    
    # axis limits from view 
    # fig_minlim_xy, fig_maxlim_xy = 0.3, 0.54
    # fig_minlim_z, fig_maxlim_z = 0, 1.4
    
    fig_minlim_xy, fig_maxlim_xy = -1, 1
    fig_minlim_z, fig_maxlim_z = 0, 1.4
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_mod, y_mod, z_mod)
    
    ax.view_init(elev=45, azim=0)
    ax.dist = 8
    ax.set_xlim([fig_minlim_xy, fig_maxlim_xy])
    ax.set_ylim([fig_minlim_xy, fig_maxlim_xy])
    ax.set_zlim([fig_minlim_z, fig_maxlim_z])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    for f in h0_1_faces:
        face = a3.art3d.Poly3DCollection([f])
        # face.set_color(mpl.colors.rgb2hex(sp.rand(3)))
        face.set_color(mpl.colors.hex2color('#787ecc'))
        face.set_edgecolor('k') 
        face.set_alpha(0.5) # transparency of faces
        ax.add_collection3d(face)
    
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    plt.show()
    plt.tight_layout()


# Global variables (Bad)
Gamma_LG = 0.076
Gamma_SG = Gamma_LG*np.cos(0.3875)
Rho_W = 997
g = 9.81


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


def Simplex_Volyme(sp, xy):
    h_ave = (sp[0, 2] + sp[1, 2] + sp[2, 2])/3
    return (xy[0]**2*h_ave)/2


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
    h_ave = (sp[0, 2] + sp[1, 2] + sp[2, 2])/3
    return (1/4)*Rho_W*g*xy[0]*xy[1]*h_ave**2


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
    a,b,c = sp[0,:], sp[1,:], sp[2,:]
    return 0.5*norm(np.cross(b-a, c-a))


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
    if (sp[0, 2] or sp[1, 2] or sp[2, 2]) > 0:
        return Gamma_LG
    else:
        return Gamma_SG


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


def System_Free_Energy(h0z, GRID_SIZE):
    """
    Calculate whole system free energy by summing
    the free energies of the simplices.

    Parameters
    ----------
    h0 : 2d-Array
        Point cloud for surface.
    h0_faces : 3d-Array
        Coordinates of the points of the simplices.

    Returns
    -------
    Float
        System free energy.

    """
    # Check right format of input data. Must be 3-dim.
    # try:
    #     len(h0[0])
    # except:
    #     h0 = np.reshape(h0, (int(len(h0)/3), 3))    
        
    h0 = Compose_h0(h0z, GRID_SIZE)
    h0_simplices = Triangulate_Data(h0)
    Delta_XY = [abs(h0[0,0] - h0[1,0]), abs(h0[0,0] - h0[1,0])]
    fe = 0
    i=0
    while i<len(h0_simplices):
        fe+=Simplex_FE(h0_simplices[i,:,:], Delta_XY)
        # Calculates Simplex volyme, seems to work
        # print(Simplex_Volyme(h0_faces[i,:,:], Delta_XY))
        i+=1
    return fe


def Compose_h0(h0z, GRID_SIZE):
    h0xy = np.load(f'FDmeshgrid_h0{GRID_SIZE}.npy')[:,:2]
    tmp=[]
    tmp.append(h0xy[:,0])
    tmp.append(h0xy[:,1])
    tmp.append(h0z)
    h0 = np.asarray(tmp).T
    return h0


def System_Volyme(h0z, GRID_SIZE):
    
    # try:
    #     len(h0[0])
    # except:
    #     h0 = np.reshape(h0, (int(len(h0)/3), 3))    
    
    h0 = Compose_h0(h0z, GRID_SIZE)
    h0_simplices = Triangulate_Data(h0)
    Delta_XY = [abs(h0[0,0] - h0[1,0]), abs(h0[0,0] - h0[1,0])]
    v = 0
    i=0
    while i<len(h0_simplices):
        v+=Simplex_Volyme(h0_simplices[i,:,:], Delta_XY)
        # Calculates Simplex volyme, seems to work
        # print(Simplex_Volyme(h0_faces[i,:,:], Delta_XY))
        i+=1
    return v


def Volyme_constrt(h0):
    return System_Volyme(h0)-0.6575252027812426 # Need to calculate for different cases


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


def Optimization_basinhopping(h0, GRID_SIZE):
    
    # bounds=[-1.,1.] , "bounds":bounds
    cons = ({'type': 'eq', 'fun': lambda h0: System_Volyme(h0, GRID_SIZE)-2.64454809049215e-10})
    minimizer_kwargs = {"method":"trust-constr", "constraints":cons, 'args':GRID_SIZE} # , "jac":True
    h0_opt = basinhopping(System_Free_Energy, h0, 
                       minimizer_kwargs=minimizer_kwargs, 
                       niter=20, 
                       T=0.1,
                       stepsize=0.00025,
                       niter_success=2,
                       interval=1,
                       disp=True,
                       target_accept_rate=0.5,
                       stepwise_factor=0.5
                       ) # , niter=200
    return h0_opt


GRID_SIZE=20
h0_1z = np.load(f'FD_NO-REST_meshgrid_h0{GRID_SIZE}.npy')[:,2]
h0_2z = np.load(f'FDmeshgrid_h0{GRID_SIZE}.npy')[:,2]
print(f"Grid size: {int(GRID_SIZE)} x {int(GRID_SIZE)}")
h0_1 = Compose_h0(h0_1z, GRID_SIZE)

# test1(h0_1)

# Test Volyme calculation
print("System volyme:      " + str(System_Volyme(h0_1z, GRID_SIZE)))
# Test Free energy calculation
print("System free energy: " + str(System_Free_Energy(h0_1z, GRID_SIZE)))

h0_opt=Optimization_basinhopping(h0_1z, GRID_SIZE)
h0_opt_z=h0_opt.x

print("Optimized System volyme:      " + str(System_Volyme(h0_opt_z, GRID_SIZE)))
print("Optimized System free energy: " + str(System_Free_Energy(h0_opt_z, GRID_SIZE)))

h0_opt_plot = Compose_h0(h0_opt_z, GRID_SIZE)

test0(h0_opt_plot)
test1(h0_opt_plot)

#%%

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
