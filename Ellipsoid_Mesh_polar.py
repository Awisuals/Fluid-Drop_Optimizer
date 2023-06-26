# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 15:58:35 2023

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
from Surfature import surface_curvature

font = {'weight' : 'normal',
        'size'   : 15}

plt.rc('font', **font)


def ellipsoid(N, abc, xyz, theta_max):
    x_points = []
    y_points = []
    z_points = []
    N = N
    a, b, c = abc[0], abc[1], abc[2]
    x0, y0, z0 = xyz[0], xyz[1], xyz[2]
    theta = np.linspace(0, theta_max, N)
    phi = np.linspace(0, 2*np.pi, N)
    def x(x0, r, theta, phi): return x0 + a*np.sin(theta)*np.cos(phi)
    def y(y0, r, theta, phi): return y0 + b*np.sin(theta)*np.sin(phi)
    def z(z0, r, theta): return z0 + c*np.cos(theta)
    
    i=0
    j=0
    while i < N:
        while j < N:
            x_points.append([x(x0, a, theta[i], phi[j])])
            y_points.append([y(y0, b, theta[i], phi[j])])
            z_points.append([z(z0, c, theta[i])])
            j+=1
        j=0
        i+=1
    return np.hstack((x_points, y_points, z_points))


# Ellipsoid params and call
N_param = 20
abc_params = [1,1,0.80]
xyz_params = [1,1,1]
theta_max = np.pi/9
points = ellipsoid(N_param, abc_params, xyz_params, theta_max)

# print(points[:,0])
x_p = points[:,0]
y_p = points[:,1]
z_p = points[:,2]

hull = sp_spatial.ConvexHull(points)
indices = hull.simplices
faces = points[indices]

print('area: ', hull.area)
print('volume: ', hull.volume)

# Figure axis scale
fig_minlim_xy, fig_maxlim_xy = 0,2
fig_minlim_z, fig_maxlim_z = 1.4, 2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=0, azim=0)
ax.dist = 8
# ax.azim = -140
ax.set_xlim([fig_minlim_xy, fig_maxlim_xy])
ax.set_ylim([fig_minlim_xy, fig_maxlim_xy])
ax.set_zlim([fig_minlim_z, fig_maxlim_z])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

for f in faces:
    face = a3.art3d.Poly3DCollection([f])
    # face.set_color(mpl.colors.rgb2hex(sp.rand(3)))
    face.set_color(mpl.colors.hex2color('#787ecc'))
    face.set_edgecolor('k') 
    face.set_alpha(1) # transparency of faces
    ax.add_collection3d(face)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()
plt.tight_layout()
# plt.close()

