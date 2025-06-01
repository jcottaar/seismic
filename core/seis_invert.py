import numpy as np
import cupy as cp
import kaggle_support as kgs
import seis_forward2
import seis_prior
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
    vec = basis_functions@x
    if compute_gradient:
        s, _, s_adjoint = seis_forward2.vel_to_seis(vec, vec_adjoint=target, adjoint_on_residual=True)
    else:
        s, _, _ = seis_forward2.vel_to_seis(vec)

    cost_residual = cp.mean( (s-target)**2 )
    if compute_gradient:
        gradient_residual = (2/len(s))*(basis_functions.T@s_adjoint)

    # Combine
    if compute_gradient:
        return cost_prior + cost_residual, gradient_prior + gradient_residual
    else:
        return cost_prior + cost_residual

true_vel = None
def seis_to_vel(seismogram, velocity_guess, prior, scaling=1e10, maxiter=2000, method='BFGS'):
    
    basis_functions = prior.basis_functions()
    x_guess = cp.asnumpy(cp.linalg.solve(basis_functions.T@basis_functions, basis_functions.T@(velocity_guess.to_vector())))
    target = seismogram.to_vector()

    # def cost_func(x):
    #     #print(x-x_guess[:,0])
    #     cost = cp.asnumpy(cost_and_gradient(cp.array(x)[:,None],target,prior,basis_functions)).item()
    #     print(cost, kgs.rms(basis_functions@cp.array(x[:,None])-true_vel.to_vector()))
    #     cost = cost*scaling
    #     return cost

    # def gradient_func(x):
    #     xx = cost_and_gradient(cp.array(x)[:,None],target,prior,basis_functions, compute_gradient=True)[1]
    #     xx = xx*scaling
    #     return cp.asnumpy(xx[:,0])

    def cost_and_gradient_func(x):
        cost,gradient = cost_and_gradient(cp.array(x)[:,None], target, prior, basis_functions, compute_gradient=True)
        if not true_vel is None:
            print(cost, kgs.rms(basis_functions@cp.array(x[:,None])-true_vel.to_vector()))
            diagnostics['vel_error_per_fev'].append(cp.asnumpy(kgs.rms(basis_functions@cp.array(x[:,None])-true_vel.to_vector())))
        cost = cost*scaling
        gradient = gradient*scaling
        return cp.asnumpy(cost), cp.asnumpy(gradient[:,0])

    #cost_func = lambda x: 
    #gradient_func = lambda x: 

    #res = scipy.optimize.minimize(cost_func, x_guess[:,0], method = 'L-BFGS-B', jac = gradient_func, options={'maxiter':maxiter})
    diagnostics = dict()
    diagnostics['vel_error_per_fev'] = []
    res = scipy.optimize.minimize(cost_and_gradient_func, x_guess[:,0], method = method, jac = True, options={'maxiter':maxiter})
    diagnostics['nit'] = res.nit
    diagnostics['nfev'] = res.nfev

    result = copy.deepcopy(velocity_guess)
    result.from_vector( basis_functions@cp.array(res.x)[:,None] )

    return result, diagnostics

@dataclass
class InversionModel(kgs.Model):
    prior: seis_prior.Prior = field(init=True, default_factory = seis_prior.RowTotalVariation)
    maxiter = 2000
    scaling = 1e15
    method = 'BFGS'

    def _infer_single(self,data):
        global true_vel
        if data.is_train:
            data.velocity.load_to_memory()
            true_vel = data.velocity
        else:
            true_vel = None
        data.velocity_guess.data = cp.array(data.velocity_guess.data)
        data.velocity_guess.min_vel = cp.array(data.velocity_guess.min_vel)
        data.velocity_guess, diagnostics = seis_to_vel(data.seismogram, data.velocity_guess, self.prior, scaling=self.scaling, maxiter=self.maxiter, method=self.method)
        data.velocity_guess.data = cp.asnumpy(data.velocity_guess.data)
        data.velocity_guess.min_vel = cp.asnumpy(data.velocity_guess.min_vel)

        data.diagnostics['seis_to_vel'] = diagnostics
        data.velocity.unload()

        return data