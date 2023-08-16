# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:47:34 2023

@author: anter

Short script to plot stuff.
"""
import matplotlib.pyplot as plt

def Plotting(xy_scale, z_scale, plot_num, plot_points, view_param, close=False):
    # Figure axis scale
    fig_minlim_xy, fig_maxlim_xy = xy_scale[0], xy_scale[1] # -1,1
    fig_minlim_z, fig_maxlim_z = z_scale[0], z_scale[1] # 0, 1.4

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Pass points to be plotted as array in func call.
    # ax.plot_surface takes 2d-arrays for x, y, z
    # for ellipsoid or plane exmaple:
    # [x, y, z_ellipsoid]
    # [x, y, z_plane]
    if plot_num == 0: ax.plot_surface(plot_points[0], 
                                      plot_points[1], 
                                      plot_points[2]) 
    # Pass points as array in func call.
    # plt.scatter takes 1d for x, y, z
    # Example ellipsoid and plane:
    # [x_points, y_points, z_points]
    if plot_num == 1: ax.scatter(plot_points[0], 
                                 plot_points[1], 
                                 plot_points[2]) # , antialiasing=True

    ax.view_init(elev=view_param[0], azim=view_param[1]) # 45, 0
    ax.dist = view_param[2] # 8
    # ax.set_xlim([fig_minlim_xy, fig_maxlim_xy])
    # ax.set_ylim([fig_minlim_xy, fig_maxlim_xy])
    ax.set_zlim([fig_minlim_z, fig_maxlim_z])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    plt.tight_layout()
    if close == True: plt.close()

