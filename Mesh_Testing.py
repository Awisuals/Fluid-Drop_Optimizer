# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:18:00 2023

@author: anter
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import numpy as np
import scipy as sp
from scipy import spatial as sp_spatial

font = {'weight' : 'normal',
        'size'   : 15}

plt.rc('font', **font)

def icosahedron():
    h = 0.5*(1+np.sqrt(5))
    p1 = np.array([[0, 1, h], [0, 1, -h], [0, -1, h], [0, -1, -h]])
    p2 = p1[:, [1, 2, 0]]
    p3 = p1[:, [2, 0, 1]]
    return np.vstack((p1, p2, p3))

def cube():
    points = np.array([
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
    ])
    return points

def sort_of_triangle():
    points = np.array([[1,0,0], [0,1,0], [0,0,1], [0,0,0]])
    return points

def random():
    rng = np.random.default_rng()
    points = rng.random((10000, 3))
    return points

def sphere():
    # points = np.array()
    x_points = []
    y_points = []
    z_points = []
    N = 20
    r = 1
    x0, y0, z0 = 0,0,0
    theta = np.linspace(0, np.pi, N) # np.pi/9
    phi = np.linspace(0, 2*np.pi, N)
    def x(x0, r, theta, phi): return x0 + r*np.sin(theta)*np.cos(phi)
    def y(y0, r, theta, phi): return y0 + r*np.sin(theta)*np.sin(phi)
    def z(z0, r, theta): return z0 + r*np.cos(theta)
    
    i=0
    j=0
    
    while i < N:
        # print(f'i on: {i}')
        while j < N:
            # print(theta[i], phi[j])
            x_points.append([x(x0, r, theta[i], phi[j])])
            y_points.append([y(y0, r, theta[i], phi[j])])
            z_points.append([z(z0, r, theta[i])])
            j+=1
        j=0
        i+=1
    return np.hstack((x_points, y_points, z_points))

def ellipsoid(N, abc, xyz, theta_max):
    # points = np.array()
    x_points = []
    y_points = []
    z_points = []
    N = N
    a, b, c = abc[0], abc[1], abc[2]
    x0, y0, z0 = xyz[0], xyz[1], xyz[2]
    theta = np.linspace(0, theta_max, N) # np.pi/9
    phi = np.linspace(0, 2*np.pi, N)
    def x(x0, r, theta, phi): return x0 + a*np.sin(theta)*np.cos(phi)
    def y(y0, r, theta, phi): return y0 + b*np.sin(theta)*np.sin(phi)
    def z(z0, r, theta): return z0 + c*np.cos(theta)
    
    i=0
    j=0
    while i < N:
        # print(f'i on: {i}')
        while j < N:
            # print(theta[i], phi[j])
            x_points.append([x(x0, a, theta[i], phi[j])])
            y_points.append([y(y0, b, theta[i], phi[j])])
            z_points.append([z(z0, c, theta[i])])
            j+=1
        j=0
        i+=1
    return np.hstack((x_points, y_points, z_points))


# points2 = icosahedron()
# points = cube()
# points = sort_of_triangle()
# points = random()
# points = sphere()

# Figure axis scale
fig_minlim_xy, fig_maxlim_xy = 0, 2
fig_minlim_z, fig_maxlim_z = 1.6, 1.9

# Ellipsoid params and call
N_param = 20
abc_params = [1,1,0.80]
xyz_params = [1,1,1]
theta_max = np.pi/9
points = ellipsoid(N_param, abc_params, xyz_params, theta_max)

hull = sp_spatial.ConvexHull(points)
indices = hull.simplices
faces = points[indices]

print('area: ', hull.area)
print('volume: ', hull.volume)


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
    face.set_alpha(1) # 0.5
    ax.add_collection3d(face)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()
plt.tight_layout()


