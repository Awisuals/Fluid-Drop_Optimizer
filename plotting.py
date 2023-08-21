# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:47:34 2023

@author: anter

Short script to plot stuff.
"""
import matplotlib.pyplot as plt
from scipy import spatial as sp_spatial
import mpl_toolkits.mplot3d as a3
import matplotlib as mpl



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

# Some plotting functions
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
