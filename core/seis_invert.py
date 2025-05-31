import numpy as np
import cupy as cp
import kaggle_support as kgs
import seis_forward
import scipy
import copy
from dataclasses import dataclass, field, fields

def cost_and_gradient(x, target, prior, basis_functions, compute_gradient=False):

    # Prior part
    if compute_gradient:
        cost_prior, gradient_prior = prior.compute_cost_and_gradient(x, compute_gradient=True)
    else:
        cost_prior = prior.compute_cost_and_gradient(x, compute_gradient=False)

    # Residual part
    vel = kgs.Velocity()
    seis = kgs.Seismogram()
    vel.from_vector(basis_functions@x)
    if compute_gradient:
        vel, JB = seis_forward.vel_to_seis(vel, seis, vel_diff_vector=basis_functions)
    else:
        vel, _ = seis_forward.vel_to_seis(vel, seis)
    v = vel.to_vector()

    cost_residual = cp.mean( (v-target)**2 )
    if compute_gradient:
        gradient_residual = 2*(JB.T)@(v-target)/len(v)    

    # Combine
    if compute_gradient:
        return cost_prior + cost_residual, gradient_prior + gradient_residual
    else:
        return cost_prior + cost_residual

def seis_to_vel(seismogram, velocity_guess, prior):
    
    basis_functions = prior.basis_functions()
    x_guess = cp.asnumpy(cp.linalg.solve(basis_functions.T@basis_functions, basis_functions.T@(velocity_guess.to_vector())))
    target = seismogram.to_vector()

    def cost_func(x):
        print(x-x_guess[:,0])
        cost = cp.asnumpy(cost_and_gradient(cp.array(x)[:,None],target,prior,basis_functions)).item()
        print(cost)
        return cost

    def gradient_func(x):
        print('gradient requested')
        xx = cost_and_gradient(cp.array(x)[:,None],target,prior,basis_functions, compute_gradient=True)[1]
        return cp.asnumpy(xx)

    #cost_func = lambda x: 
    #gradient_func = lambda x: 

    res = scipy.optimize.minimize(cost_func, x_guess[:,0], method = 'L-BFGS-B', jac = gradient_func, options={'maxiter':10})

    result = copy.deepcopy(velocity_guess)
    result.from_vector( basis_functions@cp.array(res.x)[:,None] )

    return result