# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:18:00 2023

@author: anter

This script demonstrates various 3D shape-generation methods, computes the
Convex Hull (via scipy.spatial.ConvexHull) of the resulting point cloud,
and visualizes the hull in 3D using Matplotlib.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import numpy as np
from scipy import spatial as sp_spatial

font = {'weight': 'normal',
        'size': 15}

plt.rc('font', **font)


def icosahedron():
    """
    Generate the 12 vertices of a regular icosahedron in 3D.

    Returns
    -------
    ndarray of shape (12, 3)
        The [x, y, z] coordinates of the icosahedron's vertices.
    """
    h = 0.5 * (1 + np.sqrt(5))
    p1 = np.array([[0, 1, h], [0, 1, -h], [0, -1, h], [0, -1, -h]])
    p2 = p1[:, [1, 2, 0]]
    p3 = p1[:, [2, 0, 1]]
    return np.vstack((p1, p2, p3))


def cube():
    """
    Generate the 8 corner points of a unit cube in 3D.

    Returns
    -------
    ndarray of shape (8, 3)
        The [x, y, z] coordinates of the cube's corners.
    """
    points = np.array([
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
    ])
    return points


def simple_triangle():
    """
    Generate a small tetrahedron-like set of points that form a triangular
    shape in 3D (including the origin).

    Returns
    -------
    ndarray of shape (4, 3)
        The [x, y, z] coordinates of the 4 points.
    """
    points = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])
    return points


def random_points(n=10000):
    """
    Generate a random set of n points within the unit cube [0,1]^3.

    Parameters
    ----------
    n : int, optional
        Number of random points to generate. Default is 10000.

    Returns
    -------
    ndarray of shape (n, 3)
        Random [x, y, z] points in the range [0, 1].
    """
    rng = np.random.default_rng()
    points = rng.random((n, 3))
    return points


def sphere(N=20, r=1.0, center=(0, 0, 0)):
    """
    Generate points approximating a sphere of radius r, centered at 'center',
    using a parametric grid of size N x N.

    Parameters
    ----------
    N : int
        Number of subdivisions along theta and phi.
    r : float
        Radius of the sphere.
    center : tuple of float
        (x0, y0, z0) center of the sphere.

    Returns
    -------
    ndarray of shape (N*N, 3)
        [x, y, z] coordinates covering the sphere.
    """
    x0, y0, z0 = center
    theta_vals = np.linspace(0, np.pi, N)
    phi_vals = np.linspace(0, 2 * np.pi, N)
    Theta, Phi = np.meshgrid(theta_vals, phi_vals)

    X = x0 + r * np.sin(Theta) * np.cos(Phi)
    Y = y0 + r * np.sin(Theta) * np.sin(Phi)
    Z = z0 + r * np.cos(Theta)

    return np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))


def ellipsoid(N=20, abc=(1, 1, 0.8), center=(0, 0, 0), theta_max=np.pi):
    """
    Generate points for an ellipsoid defined by semi-axes (a, b, c), with
    a maximum polar angle 'theta_max' (<= pi) to control coverage.

    Parameters
    ----------
    N : int
        Number of subdivisions for theta and phi.
    abc : tuple of float
        Semi-axes (a, b, c).
    center : tuple of float
        (x0, y0, z0) center of the ellipsoid.
    theta_max : float
        Maximum polar angle, allowing partial ellipsoid (0 < theta_max <= pi).

    Returns
    -------
    ndarray of shape (N*N, 3)
        [x, y, z] coordinates sampling the ellipsoid.
    """
    a, b, c = abc
    x0, y0, z0 = center

    theta_vals = np.linspace(0, theta_max, N)
    phi_vals = np.linspace(0, 2 * np.pi, N)
    Theta, Phi = np.meshgrid(theta_vals, phi_vals)

    X = x0 + a * np.sin(Theta) * np.cos(Phi)
    Y = y0 + b * np.sin(Theta) * np.sin(Phi)
    Z = z0 + c * np.cos(Theta)

    return np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))


if __name__ == "__main__":
    # Select which shape to generate:
    
    # points = icosahedron()
    # points = cube()
    # points = simple_triangle()
    # points = random_points(n=50)
    # points = sphere(N=10, r=1.0, center=(0, 0, 0))
    points = ellipsoid(N=20, abc=(1,1,0.80), center=(1,1,1), theta_max=np.pi/9)

    # Compute convex hull
    hull = sp_spatial.ConvexHull(points)
    indices = hull.simplices
    faces = points[indices]

    print('Hull area: ', hull.area)
    print('Hull volume: ', hull.volume)

    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=0, azim=0)
    ax.dist = 8

    # Automatically scale the axis limits based on the point data
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min(), points[:, 1].max()
    min_z, max_z = points[:, 2].min(), points[:, 2].max()
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])
    ax.set_zlim([min_z, max_z])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Plot each triangular face of the hull
    for f in faces:
        face = a3.art3d.Poly3DCollection([f])
        face.set_color(mpl.colors.hex2color('#787ecc'))
        face.set_edgecolor('k')
        face.set_alpha(1)
        ax.add_collection3d(face)

    plt.tight_layout()
    plt.show()
