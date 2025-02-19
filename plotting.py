# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:47:34 2023

@author: anter

This module provides convenient 3D plotting utilities for visualizing points,
meshes, or surfaces using Matplotlib. It offers different plot styles
(scatter, trisurf, Poly3DCollection, etc.) and optional axis scaling.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib as mpl
import numpy as np


def plot_3d_surface_or_scatter(xy_scale, z_scale, plot_mode, plot_points,
                               view_param, close=False):
    """
    Create a 3D figure and plot either a surface or scatter plot.

    NOTE: Previously named 'Plotting'.

    Parameters
    ----------
    xy_scale : tuple of float
        (xmin, xmax) for the x/y range (e.g., -1, 1). Currently this is not
        automatically applied as axis limits for x/y, but can be used if desired.
    z_scale : tuple of float
        (zmin, zmax) for z-axis range.
    plot_mode : int
        If 0, plots the data via `ax.plot_surface()`;
        If 1, plots via `ax.scatter()`.
    plot_points : list or tuple of arrays
        The data to plot, structured as needed by the chosen mode. For example:
          - `plot_points = [X, Y, Z]` for surface
          - `plot_points = [x_vals, y_vals, z_vals]` for scatter.
    view_param : tuple of (float, float, float)
        (elev, azim, dist) controlling 3D view.
    close : bool, optional
        Whether to close the figure after plotting. Defaults to False.

    Returns
    -------
    (fig, ax) : tuple
        The Matplotlib figure and 3D Axes objects, for further customization.
    """
    fig_minlim_xy, fig_maxlim_xy = xy_scale
    fig_minlim_z, fig_maxlim_z = z_scale

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if plot_mode == 0:
        # Plot a surface from 2D arrays X, Y, Z
        ax.plot_surface(plot_points[0], plot_points[1], plot_points[2])
    elif plot_mode == 1:
        # Scatter from 1D arrays x_vals, y_vals, z_vals
        ax.scatter(plot_points[0], plot_points[1], plot_points[2])
    else:
        raise ValueError("plot_mode must be 0 (surface) or 1 (scatter)")

    ax.view_init(elev=view_param[0], azim=view_param[1])
    ax.dist = view_param[2]

    # Example usage of xy_scale if you want to fix x/y:
    # ax.set_xlim([fig_minlim_xy, fig_maxlim_xy])
    # ax.set_ylim([fig_minlim_xy, fig_maxlim_xy])
    ax.set_zlim([fig_minlim_z, fig_maxlim_z])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.tight_layout()
    plt.show()

    if close:
        plt.close(fig)

    return fig, ax


def plot_scatter_points(data, ax=None, auto_scale_z=(0, 0.0025)):
    """
    Scatter plot of a Nx3 array of points (x,y,z) in a 3D axes.

    NOTE: Previously named 'test0'.

    Parameters
    ----------
    data : ndarray of shape (N, 3)
        The points to plot.
    ax : matplotlib.axes._axes.Axes, optional
        Existing 3D axes to plot into. If None, create a new figure/axes.
    auto_scale_z : tuple of float, optional
        (zmin, zmax) range for the z-axis. Defaults to (0, 0.0025).

    Returns
    -------
    (fig, ax) : tuple
        The Matplotlib figure and 3D Axes. If an existing ax was passed,
        fig may be None.
    """
    fig = None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    if auto_scale_z:
        ax.set_zlim(auto_scale_z)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if fig is not None:
        plt.tight_layout()
        plt.show()
        return fig, ax
    return None, ax


def plot_trisurf_points(data, ax=None, auto_scale_z=(0, 0.0025)):
    """
    Plot a triangular surface from scattered points in Nx3 format
    using `plot_trisurf`.

    NOTE: Previously named 'test1'.

    Parameters
    ----------
    data : ndarray of shape (N, 3)
        The points to plot in 3D.
    ax : matplotlib.axes._axes.Axes, optional
        Axes to plot into. If None, create a new figure.
    auto_scale_z : tuple of float, optional
        (zmin, zmax) range. Default (0, 0.0025).

    Returns
    -------
    (fig, ax) : tuple
        Figure and Axes used for plotting.
    """
    fig = None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    ax.plot_trisurf(data[:, 0], data[:, 1], data[:, 2], linewidth=0.2, antialiased=True)
    if auto_scale_z:
        ax.set_zlim(auto_scale_z)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if fig is not None:
        plt.tight_layout()
        plt.show()
        return fig, ax
    return None, ax


def plot_trisurf_faces(faces, ax=None):
    """
    Plot multiple triangular surfaces defined by Nx1 or Nx(...) sets of triangle
    faces, each face containing 3 points in 3D.

    NOTE: Previously named 'test2'.

    Parameters
    ----------
    faces : ndarray of shape (M, 3, 3)
        Each element is a triangle of the form [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]].
    ax : matplotlib.axes._axes.Axes, optional
        3D Axes to plot into. If None, a new figure is created.

    Returns
    -------
    (fig, ax) : tuple or (None, Axes)
        Figure and Axes used. If ax was provided, fig is None.
    """
    fig = None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    for triangle in faces:
        ax.plot_trisurf(triangle[:, 0],
                        triangle[:, 1],
                        triangle[:, 2],
                        linewidth=0.2,
                        antialiased=True)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if fig is not None:
        plt.tight_layout()
        plt.show()
        return fig, ax
    return None, ax


def plot_poly3d_faces(faces, ax=None, alpha=0.5, face_color='#787ecc'):
    """
    Use a Poly3DCollection to plot faces in 3D.

    NOTE: Previously named 'test3'.

    Parameters
    ----------
    faces : ndarray of shape (M, 3, 3)
        Each row is a triangle of 3 points in [x, y, z].
    ax : matplotlib.axes._axes.Axes, optional
        If provided, draw on this Axes. Otherwise create new figure/axes.
    alpha : float, optional
        Transparency of the triangular faces. Default 0.5.
    face_color : str, optional
        Hex color for the faces. Default '#787ecc'.

    Returns
    -------
    (fig, ax) or (None, Axes)
        Figure and Axes used for plotting. If ax was supplied, fig is None.
    """
    fig = None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    for tri in faces:
        # Create a single triangular face in 3D
        face = Poly3DCollection([tri])
        face.set_color(mpl.colors.hex2color(face_color))
        face.set_edgecolor('k')
        face.set_alpha(alpha)
        ax.add_collection3d(face)

    # Optionally auto-scale to the data
    all_points = faces.reshape(-1, 3)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if fig is not None:
        plt.tight_layout()
        plt.show()
        return fig, ax
    return None, ax
