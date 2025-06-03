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

    plt.figure()
    plt.imshow(cp.asnumpy(data.velocity.data))
    plt.colorbar()
    plt.title(data.cache_name())
    plt.pause(0.001)

    # do_which_list:
    # 0: seis_remodeled -> vel_true from vel_true
    # 1: seis_given -> vel_true from vel_true
    # 2: seis_given -> vel_true from vel_guess
    N_which = 3
    names_which = ['target=remodeled', 'target=given', 'start from guess']

    # Prep data
    vel_true = data.velocity
    seis_given = data.seismogram
    seis_remodeled = copy.deepcopy(seis_given)
    seis_remodeled.from_vector(seis_forward2.vel_to_seis(vel_true.to_vector())[0])
    vel_guess = seis_nn.default_pretrained.infer([data])[0].velocity
    vel_true_np = copy.deepcopy(vel_true)
    vel_true_np.data = cp.asnumpy(vel_true.data)

    results=[]
    for v in param_values:
        results.append([])
        print(f"{param_name}={v}")
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
            results[-1].append(model.infer([data_in]))

    
                       