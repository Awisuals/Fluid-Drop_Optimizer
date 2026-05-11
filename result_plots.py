r'''
Copyright (c) 2025 Antero Voutilainen
Created: 06 05 2026

Description: 
'''
from plotting import *
import pandas as pd

def plot_surface():
    
    G = 21
    device = 'GPU'
    
    h0_opt1 = np.load(f'FD_OPT_meshgrid_h0{G}-{device}-trustconstr.npy')
    h0_opt2 = np.load(f'FD_OPT_meshgrid_h0{G}-{device}-SLSQP.npy')
    
    # h0_eq = np.load(f'FD_REST_meshgrid_h021.npy')
    # h0_inp = np.load(f'FD_NO-REST_meshgrid_h021.npy')

    plot_trisurf_faces(data1=h0_opt2,
                    data2=h0_opt1,
                    #    data3=h0_eq,
                    z_scale=[0,0.0015],
                    view_param=[15,45,8],
                    title1="Optimized surface with SLSQP",
                    title2="Optimized surface with trust-constr"
                    #    title3="Surface representing resting droplet for comparison"
                    )    
    return


def plot_time_dependence():
    
    df = pd.read_excel('n10-30-rundata.xlsx', sheet_name='time-dependence', header=(0, 1))
    
    slsqp_cpu = df[('SLSQP', 'CPU')]
    slsqp_gpu = df[('SLSQP', 'GPU')]
    trustconstr_cpu = df[('trust-constr', 'CPU')]
    trustconstr_gpu = df[('trust-constr', 'GPU')]
    
    grids = df[('Runtime (s)', 'Grid Size NxN')]
    
    plot_multi_series(grids, 
                      [slsqp_cpu, 
                       slsqp_gpu, 
                       trustconstr_cpu, 
                       trustconstr_gpu],
                      labels=['SLSQP on CPU', 
                              'SLSQP on GPU',
                              'trust-constr on CPU', 
                              'trust-constr on GPU'],
                      xlabel='Grid size NxN',
                      ylabel='Runtime (s)',
                      title='Runtime as a function of Grid Size')
    
    # print(grids)
    
    
    return


def main():
    
    # plot_surface()
    plot_time_dependence()    
    
    return


if __name__ == '__main__':
    main()
    
    