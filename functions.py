# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:40:59 2023

@author: anter
"""
import numpy as np
from numpy.linalg import norm
from scipy.spatial import Delaunay


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
    Rho_W = 997
    g = 9.81
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
    Gamma_LG = 0.076
    Gamma_SG = Gamma_LG*np.cos(0.3875)
    if (sp[0, 2] or sp[1, 2] or sp[2, 2]) > 0.000005:
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
        
    h0 = Compose_h0(h0z, GRID_SIZE)
    h0_simplices = Triangulate_Data(h0)
    Delta_XY = [abs(h0[0,0] - h0[1,0]), abs(h0[0,0] - h0[1,0])]
    fe = 0
    i=0
    while i<len(h0_simplices):
        fe+=Simplex_FE(h0_simplices[i,:,:], Delta_XY)
        i+=1
    return fe


def Compose_h0(h0z, GRID_SIZE):
    h0xy = np.load(f'FD_REST_meshgrid_h0{GRID_SIZE}.npy')[:,:2]
    tmp=[]
    tmp.append(h0xy[:,0])
    tmp.append(h0xy[:,1])
    tmp.append(h0z)
    h0 = np.asarray(tmp).T
    return h0


def System_Volyme(h0z, GRID_SIZE):
    
    h0 = Compose_h0(h0z, GRID_SIZE)
    h0_simplices = Triangulate_Data(h0)
    Delta_XY = [abs(h0[0,0] - h0[1,0]), abs(h0[0,0] - h0[1,0])]
    v = 0
    i=0
    while i<len(h0_simplices):
        v+=Simplex_Volyme(h0_simplices[i,:,:], Delta_XY)
        i+=1
    return v


def Find_Grid_Boundary(grid_data):
    
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


def Epsilon(value, epsilon=1e-8):
    if value > epsilon:
        return value
    else:
        return epsilon


def find_indices_greater_than_zero(arr):
    return np.where(arr > 0)[0]


