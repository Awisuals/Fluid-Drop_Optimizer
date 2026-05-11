r'''
Copyright (c) 2025 Antero Voutilainen
Created: 05 05 2026

Description: 
'''
import numpy as np


def compute_errors(GRID_SIZE : int,
                   data_selection : int) -> None:
    
    GRID = GRID_SIZE
    
    opt_names = ['GPU-trustconstr', 
                'GPU-SLSQP', 
                'CPU-trustconstr', 
                'CPU-SLSQP']
    
    opt_name = opt_names[data_selection]
    
    # Import data, OPT and EQ
    z_opt = np.load(f'FD_OPT_meshgrid_h0{GRID}-{opt_name}.npy')[:, 2]
    z_eq = np.load(f'FD_REST_meshgrid_h0{GRID}.npy')[:, 2]
    
    # Compute l2 error
    # shape_error = 100* np.linalg.norm(z_opt - z_eq) / np.linalg.norm(z_eq)
    # print(shape_error)
    
    # Compute RMSE
    rmse = 100 * np.sqrt(np.mean((z_opt - z_eq)**2)) / np.max(z_eq)
    print(rmse)
    
    return

def main():
    
    data_selection = [0, 1, 2, 3]
    
    GRID_SIZE = [10,11,12,13,14,15,16,17,18,19,20,
                 21,22,23,24,25,26,27,28,29,30]
    
    for selection in data_selection:
        print('\n\n')    
        for each in GRID_SIZE:
            compute_errors(each, selection)
    
    
    return


if __name__ == '__main__':
    main()