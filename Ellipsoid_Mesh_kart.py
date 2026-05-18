# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:07:04 2023

@author: anter
"""
import numpy as np
import math as m
from plotting import *
from surfature import Surface_Curvature
from functions import *

def ellipsoid_z(X, Y, a, b, c):
    # z = np.sqrt(c**2*(1-(X**2/a**2)-(Y**2/b**2)))# -0.5
    z = (c+b*np.sqrt(1-a*(X**2+Y**2)))
    for idr, row in enumerate(z):
        for idc, column in enumerate(row):
            if m.isnan(column):
                # print(column)
                z[idr,idc] = 0
    return z

def plane_z(X, Y, a, b, c, d):
    return (a*X+b*Y+d)/(c)-0.5

def remove_plane_points(X, Y, Z, height):
    """
    Define height which removes points from z array 
    and corresponding points from x, y arrays.

    Parameters
    ----------
    X : Array
        1d-array of x-points.
    Y : Array
        1d-array of y-points.
    Z : Array
        1d-array of z-points.
    height : Float
        Height to be remved from arrays.

    Returns
    -------
    Modified arrays.

    """
    Z_mod = []
    X_mod = []
    Y_mod = []
    print("Z-taulukko koko")
    print(len(Z))
    for ide, emb in enumerate(Z):
        if 0 < ide < len(Z)-1:
            if height != emb or (emb == height and (Z[ide+1] != height or Z[ide-1] != height)):
                Z_mod.append(Z[ide])
                X_mod.append(X[ide])
                Y_mod.append(Y[ide])
            
                # Z_mod = np.delete(Z, [ide])
                # X_mod = np.delete(X, [ide])
                # Y_mod = np.delete(Y, [ide])
    
    return [X_mod, Y_mod, Z_mod]

def generate_grids(max_grid_size):
    i=10
    while i<max_grid_size:
        GRID_SCALE=0.0035
        N = i
        plane_params = [0, 0, 2, 1]
        x = np.linspace(-GRID_SCALE,GRID_SCALE,N)
        y = np.linspace(-GRID_SCALE,GRID_SCALE,N)
        [x, y] = np.meshgrid(x,y, sparse=False)
        z_ellipsoid = ellipsoid_z(x, y, 1.5e+05, # 5.846411733602286e+04 
                                  0.000000290055701e+04, 
                                  -0.000000250719341e+04) #0.75, 0.75, 0.8)
        z_plane = plane_z(x, y, plane_params[0], plane_params[1], 
                  plane_params[2], plane_params[3])
        def Cut_Ellipsoid_With_Plane():    
            for idr, r in enumerate(z_ellipsoid):
                for idc, c in enumerate(r):
                    if z_plane[idr, idc] > z_ellipsoid[idr, idc]:
                        z_ellipsoid[idr, idc] = z_plane[idr, idc]
        Cut_Ellipsoid_With_Plane()
        
        x_points = np.reshape(x, -1)
        y_points = np.reshape(y, -1)
        z_points = np.reshape(z_ellipsoid, -1)
        ellipsoid_points = np.stack((x_points, y_points, z_points), 1)
        
        np.save('FD_NO-REST_meshgrid_h0'+str(i), ellipsoid_points)
        i+=1


def Cut_Ellipsoid_With_Plane(ellipsoid, plane):    
    for idr, r in enumerate(ellipsoid):
        for idc, c in enumerate(r):
            if plane[idr, idc] > ellipsoid[idr, idc]:
                ellipsoid[idr, idc] = plane[idr, idc]
    
# generate_grids(31)

GRID_SCALE=0.0035
# GRID_SCALE=0.05
N = 20
plane_params = [0, 0, 2, 1]
x = np.linspace(-GRID_SCALE,GRID_SCALE,N)
y = np.linspace(-GRID_SCALE,GRID_SCALE,N)
[x, y] = np.meshgrid(x,y, sparse=False)

# ==========================================================================

z_ellipsoid_rest = ellipsoid_z(x, y, 5.846411733602286e+04, # 5.846411733602286e+04 
                          0.000000290055701e+04, 
                          -0.000000250719341e+04) #0.75, 0.75, 0.8)
z_plane = plane_z(x, y, plane_params[0], plane_params[1], 
          plane_params[2], plane_params[3])
Cut_Ellipsoid_With_Plane(z_ellipsoid_rest, z_plane)

x_points = np.reshape(x, -1)
y_points = np.reshape(y, -1)
z_points_rest = np.reshape(z_ellipsoid_rest, -1)
ellipsoid_points_rest = np.stack((x_points, y_points, z_points_rest), 1)

# ===========================================================================

# Create norest ellipsoid
# Multiply z-values with const 2.56... (based on wrong assumptions but kinda works)
z_ellipsoid_norest = 2.5540611332565573 * ellipsoid_z(x, y, 1.5e+05, # 2.5540611332565573 *   
                          0.000000290055701e+04, 
                          -0.000000250719341e+04) #0.75, 0.75, 0.8)

z_plane = plane_z(x, y, plane_params[0], plane_params[1], 
          plane_params[2], plane_params[3])
Cut_Ellipsoid_With_Plane(z_ellipsoid_norest, z_plane)

x_points = np.reshape(x, -1)
y_points = np.reshape(y, -1)
z_points_norest = np.reshape(z_ellipsoid_norest, -1)
ellipsoid_points_norest = np.stack((x_points, y_points, z_points_norest), 1)

# ellipsoid_points_norest = np.load(f'FD_NO-REST_meshgrid_h0{N}.npy')
# print(ellipsoid_points_norest)

# np.save('FD_NO-REST_meshgrid_h0'+str(N), ellipsoid_points)
# np.save('FD_REST_meshgrid_h0'+str(N), ellipsoid_points)

# Test Volyme calculation
print("No Rest System volyme: " + str(system_volume(ellipsoid_points_norest[:, 2], N)))
print("Rest System volyme: " + str(system_volume(ellipsoid_points_rest[:, 2], N)))

# Test Energy calculation
print("No Rest System energy: " + str(system_free_energy(ellipsoid_points_norest[:, 2], N)))
print("Rest System energy: " + str(system_free_energy(ellipsoid_points_rest[:, 2], N)))

plot_3d_surface_or_scatter([0,0.0015], 
                           1, 
                           [x,y,z_ellipsoid_rest], 
                           [25,45,8], 
                           title="Point cloud representation of a sessile droplet")

plot_trisurf_faces(data1=ellipsoid_points_norest,
                   data2=ellipsoid_points_rest,
                   z_scale=[0,0.0015],
                   view_param=[15,45,8],
                   title1='Surface representinig modified droplet input',
                   title2='Surface representing resting droplet for comparison'
                #    title="Triangulated surface representation of a sessile droplet in equilibrium"
                   )

