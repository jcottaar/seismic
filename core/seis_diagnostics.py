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
    data.load_to_memory()

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
    seis_remodeled.from_vector(seis_forward2.vel_to_seis(vel_true.to_vector()))
    vel_guess = seis_nn.make_default_pretrained().infer([d])[0].velocity

    seis_forward2.true_vel = data.velocity

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
                    seis = seis_remodeled
                    vel_start = vel_true
                case 1:
                    seis = seis_given
                    vel_start = vel_true
                case 2:
                    seis = seis_given
                    vel_start = vel_guess
                    
                    

    seis_forward2.true_vel = None
        

    
                       