# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:40:59 2023

@author: anter

This file comprises a set of computational routines for the advanced modeling
of droplet surfaces on a discretized grid. The methods herein facilitate:

1) Triangulation of three-dimensional point sets (x, y, z) into two-dimensional
   simplices using Delaunay tessellation.
2) Quantitative evaluation of volumetric, surface tension, and gravitational
   contributions to the droplet’s total free energy.
3) Identification of boundary indices in a square or rectangular grid,
   supporting the imposition of boundary conditions in droplet simulations.

Physics Model Rationale:
- Each droplet is approximated by aggregating small triangular "beams." Volume
  and energy are computed by averaging nodal heights (z-values) within each
  triangle and applying a local 2D width (Delta_XY) for scaling.
- The code distinguishes between liquid-gas and solid-gas surface tensions,
  based on a specified critical threshold for the z-coordinate.
- Gravitational energy for each triangular element is computed through the
  average height of that simplex, multiplied by relevant constants such as
  fluid density and gravitational acceleration.
"""

import numpy as np
from numpy.linalg import norm
from scipy.spatial import Delaunay

# -------------------
# Named constants for droplet modeling:
# -------------------
LIQUID_GAS_TENSION = 0.076   # [N/m] Nominal surface tension for water-air
CONTACT_ANGLE = 0.3875       # [radians] Representative contact angle
CRITICAL_HEIGHT = 0.000005   # [m] Threshold for distinguishing liquid-gas vs. solid-gas
RHO_WATER = 997              # [kg/m^3] Density of water
GRAVITY = 9.81               # [m/s^2] Gravitational acceleration


def triangulate_data(data):
    """
    Execute a Delaunay-based triangulation over the (x, y) coordinates of a 3D
    point cloud, returning simplices that each comprises three (x, y, z) vertices.

    Parameters
    ----------
    data : ndarray of shape (N, 3)
        A set of N points in R^3, where each row represents [x, y, z].

    Returns
    -------
    data_simplices : ndarray of shape (M, 3, 3)
        The triangulated simplices, each containing three [x, y, z] vertices.
    """
    data_tri = Delaunay(data[:, :2])
    data_indices = data_tri.simplices
    data_simplices = data[data_indices]
    return data_simplices


def simplex_volume(simplex, delta_xy):
    """
    Compute the volumetric contribution of a single triangular "beam." This
    approximation leverages an averaged z-value of the simplex’s vertices.

    Parameters
    ----------
    simplex : ndarray of shape (3, 3)
        Coordinates of the triangle’s corners, each [x, y, z].
    delta_xy : tuple or list of length 2
        (delta_x, delta_y) reflecting the local spacing in x and y.

    Returns
    -------
    float
        Estimated volume associated with the simplex.
    """
    h_ave = (simplex[0, 2] + simplex[1, 2] + simplex[2, 2]) / 3.0
    # Example formula: (delta_x^2 * h_ave) / 2
    return (delta_xy[0] ** 2 * h_ave) / 2.0


def simplex_gravitational_energy(simplex, delta_xy):
    """
    Calculate the gravitational potential energy of a triangular segment.
    This metric is computed by referencing the average height of its vertices.

    Parameters
    ----------
    simplex : ndarray of shape (3, 3)
        [x, y, z] coordinates defining the triangle.
    delta_xy : tuple or list (delta_x, delta_y)
        Characteristic lengths in x and y.

    Returns
    -------
    float
        Gravitational energy for this simplex.
    """
    h_ave = (simplex[0, 2] + simplex[1, 2] + simplex[2, 2]) / 3.0
    return 0.25 * RHO_WATER * GRAVITY * delta_xy[0] * delta_xy[1] * (h_ave ** 2)


def simplex_area(simplex):
    """
    Determine the area of a triangle embedded in three-dimensional space,
    employing the cross product of its edges.

    Parameters
    ----------
    simplex : ndarray of shape (3, 3)
        Three [x, y, z] points defining the triangle.

    Returns
    -------
    float
        The computed surface area of the triangle.
    """
    a, b, c = simplex[0, :], simplex[1, :], simplex[2, :]
    return 0.5 * norm(np.cross(b - a, c - a))


def simplex_surface_tension(simplex):
    """
    Select the appropriate surface tension coefficient based on whether
    any vertex’s z-value surpasses a designated critical height.

    Parameters
    ----------
    simplex : ndarray of shape (3, 3)
        Coordinates of the triangle’s three vertices.

    Returns
    -------
    float
        Liquid-gas surface tension if above CRITICAL_HEIGHT; else solid-gas.
    """
    gamma_sg = LIQUID_GAS_TENSION * np.cos(CONTACT_ANGLE)
    if np.any(simplex[:, 2] > CRITICAL_HEIGHT):
        return LIQUID_GAS_TENSION
    else:
        return gamma_sg


def simplex_free_energy(simplex, delta_xy):
    """
    Compute the total free energy of a triangular region by summing the surface
    tension term and its gravitational contribution.

    Parameters
    ----------
    simplex : ndarray of shape (3, 3)
        [x, y, z] coordinates of the triangle.
    delta_xy : tuple or list of length 2
        Local mesh spacing (delta_x, delta_y).

    Returns
    -------
    float
        Combined surface tension and gravitational energy for the simplex.
    """
    st_coeff = simplex_surface_tension(simplex)
    area = simplex_area(simplex)
    ge = simplex_gravitational_energy(simplex, delta_xy)
    return st_coeff * area + ge


def get_delta_xy(points):
    """
    Derive approximate x and y spacings (delta_x, delta_y) from the initial
    pair of points in the dataset. This approach assumes uniform grid spacing.

    Parameters
    ----------
    points : ndarray of shape (N, 3)
        Collection of [x, y, z] coordinates for the droplet.

    Returns
    -------
    tuple
        (delta_x, delta_y)
    """
    delta_x = abs(points[0, 0] - points[1, 0])
    delta_y = abs(points[0, 1] - points[1, 1])
    return (delta_x, delta_y)


def system_free_energy(h0z, grid_size, xy_data=None):
    """
    Assemble the droplet’s 3D representation from z-values, apply Delaunay
    triangulation, and integrate each simplex’s free energy.

    Parameters
    ----------
    h0z : ndarray
        Array of z-values representing droplet height.
    grid_size : int
        Dimension of the grid in use.
    xy_data : ndarray of shape (N, 2), optional
        User-provided (x, y) coordinates. If omitted, a default file is loaded.

    Returns
    -------
    float
        Aggregate free energy of the droplet’s discretized surface.
    """
    points = compose_h0(h0z, grid_size, xy_data)
    simplices = triangulate_data(points)
    delta_xy = get_delta_xy(points)

    total_fe = 0.0
    for simplex in simplices:
        total_fe += simplex_free_energy(simplex, delta_xy)
    return total_fe


def compose_h0(h0z, grid_size, xy_data=None):
    """
    Construct an Nx3 array of [x, y, z] coordinates by merging separate
    z-values with either supplied x,y data or a fallback file.

    Parameters
    ----------
    h0z : ndarray of shape (N,)
        Z-values of the droplet.
    grid_size : int
        Dimension of the grid.
    xy_data : ndarray of shape (N, 2), optional
        If specified, these coordinates are used directly.

    Returns
    -------
    ndarray
        Concatenated array of [x, y, z] for subsequent processing.
    """
    if xy_data is None:
        h0xy = np.load(f'FD_REST_meshgrid_h0{grid_size}.npy')[:, :2]
    else:
        h0xy = xy_data

    combined = np.column_stack((h0xy[:, 0], h0xy[:, 1], h0z))
    return combined


def system_volume(h0z, grid_size, xy_data=None):
    """
    Sum the volumetric contributions of each triangular element to approximate
    the droplet’s total volume.

    Parameters
    ----------
    h0z : ndarray
        Array of z-values defining droplet height.
    grid_size : int
        Grid dimension for reconstructing x,y.
    xy_data : ndarray of shape (N, 2), optional
        If given, these (x, y) coordinates are utilized directly.

    Returns
    -------
    float
        Integrated volume of the droplet.
    """
    points = compose_h0(h0z, grid_size, xy_data)
    simplices = triangulate_data(points)
    delta_xy = get_delta_xy(points)

    total_vol = 0.0
    for simplex in simplices:
        total_vol += simplex_volume(simplex, delta_xy)
    return total_vol


def find_grid_boundary(grid_data):
    """
    Determine the boundary indices of a square or rectangular grid by locating
    the extremal rows (top/bottom) and columns (left/right).

    Parameters
    ----------
    grid_data : ndarray of shape (N, 3) or (N, 2)
        Points representing the grid, typically [x, y, (z)].

    Returns
    -------
    ndarray
        Sorted array of unique boundary indices.
    """
    unique_y_values = np.unique(grid_data[:, 1])
    unique_x_values = np.unique(grid_data[:, 0])

    num_rows = len(unique_y_values)
    num_columns = len(unique_x_values)

    top_left_corner = 0
    top_right_corner = num_columns - 1
    bottom_left_corner = (num_rows - 1) * num_columns
    bottom_right_corner = num_rows * num_columns - 1

    top_side_indices = np.arange(top_left_corner, top_right_corner + 1)
    bottom_side_indices = np.arange(bottom_left_corner, bottom_right_corner + 1)
    left_side_indices = np.arange(top_left_corner, bottom_left_corner + 1, num_columns)
    right_side_indices = np.arange(top_right_corner, bottom_right_corner + 1, num_columns)

    all_side_indices = np.concatenate([
        top_side_indices,
        bottom_side_indices,
        left_side_indices,
        right_side_indices
    ])

    boundary_unique_values = np.unique(all_side_indices)
    return boundary_unique_values


def epsilon_value(value, eps=1e-8):
    """
    Impose a lower bound on a floating-point value to avoid pathological
    scenarios (e.g., division by zero).

    Parameters
    ----------
    value : float
        Input value.
    eps : float, optional
        Minimum permissible value.

    Returns
    -------
    float
        Either the original value if it is larger than eps, or eps otherwise.
    """
    return value if value > eps else eps


def find_indices_greater_than_zero(arr):
    """
    Locate the indices of any array elements that exceed zero.

    Parameters
    ----------
    arr : ndarray
        The array to be examined.

    Returns
    -------
    ndarray of int
        Positions in arr where values are greater than zero.
    """
    return np.where(arr > 0)[0]
