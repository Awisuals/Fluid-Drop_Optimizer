# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:07:04 2023

@author: anter
"""
import pylab as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import numpy as np
import scipy as sp
import math as m
from scipy import spatial as sp_spatial
from plotting import Plotting
from surfature import Surface_Curvature
from scipy.spatial import Delaunay
from functions import *

def ellipsoid_z(X, Y, a, b, c):
    # z = np.sqrt(c**2*(1-(X**2/a**2)-(Y**2/b**2)))# -0.5
    z = (c+b*np.sqrt(1-a*(X**2+Y**2)))*2.5657562917683223
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
    
# generate_grids(31)

GRID_SCALE=0.0035
N = 100
plane_params = [0, 0, 2, 1]
x = np.linspace(-GRID_SCALE,GRID_SCALE,N)
y = np.linspace(-GRID_SCALE,GRID_SCALE,N)
[x, y] = np.meshgrid(x,y, sparse=False)
z_ellipsoid = ellipsoid_z(x, y, 1.5e+05, # 1.5e+05 5.846411733602286e+04 
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

# np.save('FD_NO-REST_meshgrid_h0'+str(N), ellipsoid_points)
# np.save('FD_REST_meshgrid_h0'+str(N), ellipsoid_points)

# Test Volyme calculation
print("System volyme:      " + str(System_Volyme(z_points, N)))

Plotting([-1,1], [0,0.0025], 0, 
          [x,y,z_ellipsoid], [45,0,8], close=True)


# =============================================================


# Modify points so we have just ellipsoid
x_mod, y_mod, z_mod = remove_plane_points(x_points, 
                                          y_points, 
                                          z_points, 
                                          z_points[0])
ellipsoid_points_mod = np.stack((x_mod, y_mod, z_mod), 1)

# print(f'z_points first value = {z_points[0]}')

# Reshape points to 2d array
# x_2d = np.reshape(x_mod, (N, N))
# y_2d = np.reshape(y_mod, (N, N))
# z_2d = np.reshape(z_mod, (N, N))
# ellipsoid_points = np.stack((x_2d, y_2d, z_2d), 1)

# Plotting just points
Plotting([-1,1], [-1, 1], 1,# for z 0,1.4
         [x_points,y_points,z_points], [45,0,8], 
         close=True)

Plotting([-1,1], [-1,1], 1, 
         [x_mod,y_mod,z_mod], [45,0,8], 
         close=True)

# Plotting([-1,1], [0,1.4], 0, 
#          [x_2d, y_2d, z_2d], [45,0,8], 7
#          close=False)

# Calculate surfature
# surfature = Surface_Curvature(x, y, z_ellipsoid)
# print(f'Max surfature: {surfature[0]}')
# print(f'Min surfature: {surfature[1]}')
# max_surfature = np.reshape(surfature[0], (N, N))
# min_surfature = np.reshape(surfature[1], (N, N))


# Mesh from points
# hull = Delaunay(ellipsoid_points_mod[:,:2]) # Takes all dimensions, reduce to two to work?
# indices = hull.simplices
# faces = ellipsoid_points_mod[indices] # REMEMBER THIS AS WELL!!

# print('area: ', hull.area)
# print('volume: ', hull.volume)

# axis limits from view 
# fig_minlim_xy, fig_maxlim_xy = 0.3, 0.54
# fig_minlim_z, fig_maxlim_z = 0, 1.4

# fig_minlim_xy, fig_maxlim_xy = -1, 1
# fig_minlim_z, fig_maxlim_z = 0, 1.4

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(x_mod, y_mod, z_mod)

# ax.view_init(elev=45, azim=0)
# ax.dist = 8
# ax.set_xlim([fig_minlim_xy, fig_maxlim_xy])
# ax.set_ylim([fig_minlim_xy, fig_maxlim_xy])
# ax.set_zlim([fig_minlim_z, fig_maxlim_z])
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

# for f in faces:
#     face = a3.art3d.Poly3DCollection([f])
#     # face.set_color(mpl.colors.rgb2hex(sp.rand(3)))
#     face.set_color(mpl.colors.hex2color('#787ecc'))
#     face.set_edgecolor('k') 
#     face.set_alpha(1) # transparency of faces
#     ax.add_collection3d(face)

# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
# plt.show()
# plt.tight_layout()
# plt.close()
