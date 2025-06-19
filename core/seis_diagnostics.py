import numpy as np
import cupy as cp
import kaggle_support as kgs
import seis_forward2
import seis_prior
import seis_invert
import seis_nn
import scipy
import copy
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass, field, fields
from matplotlib import animation, rc; rc('animation', html='jshtml')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 1000
matplotlib.rcParams['animation.html'] = 'jshtml'

def do_diagnostics_run(data, model, do_which_list, param_func, param_values, param_name, help_fac=0.0, start_model = seis_nn.default_pretrained):
    model.check_constraints()
    model.run_in_parallel = False
    model.train([],[])
    data.load_to_memory()

    _,ax = plt.subplots(1,2,figsize=(10,4))
    plt.sca(ax[0])
    plt.imshow(cp.asnumpy(data.velocity.data))
    plt.colorbar()
    plt.title(data.cache_name())


    # do_which_list:
    # 0: seis_remodeled -> vel_true from vel_true
    # 1: seis_given -> vel_true from vel_true
    # 2: seis_given -> vel_true from vel_guess
    N_which = 3
    names_which = ['target=remodeled', 'target=given', 'start from guess']
    plot_names = ['velocity MAE error', 'seismogram RMS error', 'total cost'] 
    plot_fields = ['vel_error_per_fev', 'seis_error_per_fev', 'total_cost_per_fev']

    kgs.disable_caching = False
    # Prep data
    vel_true = data.velocity
    seis_given = data.seismogram
    seis_remodeled = copy.deepcopy(seis_given)
    seis_remodeled.from_vector(seis_forward2.vel_to_seis(vel_true.to_vector())[0])
    vel_guess = start_model.infer([data])[0].velocity_guess
    vel_true_np = copy.deepcopy(vel_true)
    vel_true_np.data = cp.asnumpy(vel_true.data)
    kgs.disable_caching = True

    plt.sca(ax[1])
    plt.imshow(vel_guess.data - cp.asnumpy(data.velocity.data))
    plt.colorbar()
    plt.title('Guess error')
    plt.pause(0.001)

    results=[]
    for v in param_values:
        results.append([])
        #print(f"{param_name}={v}")
        plt.pause(0.001)
        this_model = param_func(copy.deepcopy(model), v)
        

        
        names = []
        for i_which in range(len(do_which_list)):            
            if not do_which_list[i_which]:
                continue
            match i_which:
                case 0:
                    seis_target = seis_remodeled
                    vel_start = vel_true_np
                case 1:
                    seis_target = seis_given
                    vel_start = vel_true_np
                case 2:
                    seis_target = seis_given
                    vel_start = vel_guess
                    vel_start.data = help_fac*vel_true_np.data + (1-help_fac)*vel_guess.data                  
                    vel_start.min_vel = help_fac*vel_true_np.min_vel + (1-help_fac)*vel_guess.min_vel 
            print('diag', kgs.rms(seis_target.to_vector()- seis_forward2.vel_to_seis(vel_true.to_vector())[0]))
            data_in = copy.deepcopy(data)
            data_in.velocity_guess = vel_start
            data_in.seismogram = seis_target
            results[-1].append(copy.deepcopy(this_model.infer([data_in])[0]))
            names.append(names_which[i_which])

        
        # Plot convergence behavior for this batch
        _,ax = plt.subplots(1,3,figsize=(15,4))
        for i_plot in range(3):
            plt.sca(ax[i_plot])
            plt.xlabel('Time [s]')
            plt.ylabel(plot_names[i_plot])
            plt.grid(True)
            for rr in results[-1]:
                plt.semilogy(rr.diagnostics['seis_to_vel']['time_per_fev'], rr.diagnostics['seis_to_vel'][plot_fields[i_plot]])            
        plt.legend(names)
        plt.suptitle(f"{param_name}={v}")

        _,ax = plt.subplots(1,len(names),figsize=(5*len(names),4))
        for i_plot in range(len(names)):
            if len(names)>1:
                plt.sca(ax[i_plot])
            plt.imshow(results[-1][i_plot].velocity_guess.data-vel_true_np.data)
            plt.colorbar()
            plt.title(names[i_plot])
        plt.suptitle(f"{param_name}={v}")
        
        plt.pause(0.001)

    # Plot convergence behavior for all batches
    _,ax = plt.subplots(1,3,figsize=(15,4))
    for i_plot in range(3):
        plt.sca(ax[i_plot])
        plt.xlabel('Time [s]')
        plt.ylabel(plot_names[i_plot])
        plt.grid(True)
        for rr in results:
            plt.semilogy(rr[-1].diagnostics['seis_to_vel']['time_per_fev'], rr[-1].diagnostics['seis_to_vel'][plot_fields[i_plot]])            
    plt.legend(param_values)
    plt.title(names[-1])

    # Plot results for all batches
    _,ax = plt.subplots(1,3,figsize=(15,4))
    for i_plot in range(3):
        plt.sca(ax[i_plot])
        plt.xlabel(param_name)
        plt.ylabel(plot_names[i_plot])
        plt.grid(True)
        for i_result in range(len(results[-1])):
            y_vals = [r[i_result].diagnostics['seis_to_vel'][plot_fields[i_plot]][-1] for r in results]
            plt.semilogy(param_values, y_vals)  
    plt.legend(names)

    return results
            

    

def animate_3d_matrix(animation_arr, fps=20, figsize=(6,6), axis_off=False):

    animation_arr= copy.deepcopy(animation_arr[...])
    
    # Initialise plot
    fig = plt.figure(figsize=figsize)  # if size is too big then gif gets truncated

    im = plt.imshow(animation_arr[0])
    plt.clim([0, 1])
    if axis_off:
        plt.axis('off')
    #plt.title(f"{tomo_id}", fontweight="bold")

    min_val = np.percentile(animation_arr, 2)
    max_val = np.percentile(animation_arr,98)
    print('range: ', min_val,max_val)
    animation_arr = (animation_arr-min_val)/(max_val-min_val)
    # Load next frame
    def animate_func(i):
        im.set_data(animation_arr[i])
        #plt.clim([0, 1])
        return [im]
    plt.close()
    
    # Animation function
    anim = animation.FuncAnimation(fig, animate_func, frames = animation_arr.shape[0], interval = 1000//fps, blit=True)

    display(anim)
    return