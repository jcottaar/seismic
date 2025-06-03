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
from dataclasses import dataclass, field, fields

def do_diagnostics_run(data, model, do_which_list, param_func, param_values, param_name):
    model.check_constraints()
    model.run_in_parallel = False
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
    plot_names = ['velocity RMS error', 'seismogram RMS error', 'prior cost'] 
    plot_fields = ['vel_error_per_fev', 'seis_error_per_fev', 'prior_cost_per_fev']

    # Prep data
    vel_true = data.velocity
    seis_given = data.seismogram
    seis_remodeled = copy.deepcopy(seis_given)
    seis_remodeled.from_vector(seis_forward2.vel_to_seis(vel_true.to_vector())[0])
    vel_guess = seis_nn.default_pretrained.infer([data])[0].velocity_guess
    vel_true_np = copy.deepcopy(vel_true)
    vel_true_np.data = cp.asnumpy(vel_true.data)

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
            data_in = copy.deepcopy(data)
            data_in.velocity_guess = vel_start
            data_in.seismogram = seis_target
            results[-1].append(copy.deepcopy(this_model.infer([data_in])[0]))
            names.append(names_which[i_which])

        # Plot convergence behavior for this batch
        _,ax = plt.subplots(1,3,figsize=(15,4))
        for i_plot in range(3):
            plt.sca(ax[i_plot])
            plt.xlabel('Function evaluations')
            plt.ylabel(plot_names[i_plot])
            plt.grid(True)
            for rr in results[-1]:
                plt.semilogy(rr.diagnostics['seis_to_vel'][plot_fields[i_plot]])            
        plt.legend(names)
        plt.suptitle(f"{param_name}={v}")

        _,ax = plt.subplots(1,len(names),figsize=(5*len(names),4))
        for i_plot in range(len(names)):
            plt.sca(ax[i_plot])
            plt.imshow(results[-1][i_plot].velocity_guess.data-vel_true_np.data)
            plt.colorbar()
            plt.title(names[i_plot])
        plt.suptitle(f"{param_name}={v}")
        
        plt.pause(0.001)

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
            

    
                       