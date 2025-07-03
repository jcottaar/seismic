'''
Implements the core algorithms for full waveform inversion.
'''


import numpy as np
import cupy as cp
import kaggle_support as kgs
import seis_forward2
import seis_prior
import seis_numerics
import scipy
import copy
from dataclasses import dataclass, field, fields
import torch
from torch.utils.dlpack import to_dlpack, from_dlpack

def cost_and_gradient(x, target, prior, basis_functions, compute_gradient=False):
# Function to compute our overall cost function and its gradient.
# Cost function for velocity map x is: p(x) + ||s(x)-t||_2^2/N
# p(x) is the penalty function defined by the prior
# s(x) is the seismogram that would be measured from x
# t is the target seismogram
# N is the number of values in the seismogram

    # Compute p(x)
    if compute_gradient:
        cost_prior, gradient_prior = prior.compute_cost_and_gradient(x, compute_gradient=True)
    else:
        cost_prior = prior.compute_cost_and_gradient(x, compute_gradient=False)

    # Compute s(x)
    vec = basis_functions@x
    if compute_gradient:
        s, _, s_adjoint = seis_forward2.vel_to_seis(vec, vec_adjoint=target, adjoint_on_residual=True)
    else:
        s, _, _ = seis_forward2.vel_to_seis(vec)

    # Compute ||s(x)-t||_2^2/N
    cost_residual = cp.mean( (s-target)**2 )
    if compute_gradient:
        gradient_residual = (2/len(s))*(basis_functions.T@s_adjoint)

    # Combine
    if compute_gradient:
        return cost_prior + cost_residual, gradient_prior + gradient_residual, cost_prior, cost_residual
    else:
        return cost_prior + cost_residual, cost_prior, cost_residual


@dataclass
class InversionModel(kgs.Model):
    prior: seis_prior.Prior = field(init=True, default_factory = seis_prior.RowTotalVariation)
    # Prior definition
    
    history_size = 10000 # History size for LBFGS
    scaling = 1e15 # Scaling applied to cost function (helps in numerics)
    lbfgs_tolerance_grad = 1e-7 # Stopping criterion for LBFGS
    lbfgs_tolerance_change = 0.0 # Stopping criterion for LBFGS
    seis_error_tolerance = 0.0 # Stopping criterion for LBFGS (based on MSE of seismogram error)

    _prior_in_use = 0 # Used internally to do some internal computations in the prior
    
    iter_list = 0

    def seis_to_vel_gn(self, seismogram, velocity_guess):
        basis_functions = self._prior_in_use.basis_vectors
        
        x_guess = cp.linalg.solve(cp.array(basis_functions.T@basis_functions), basis_functions.T@(velocity_guess.to_vector()))
        x_guess = x_guess.astype(dtype=kgs.base_type)
        target = seismogram.to_vector()
        vec = velocity_guess.to_vector()

        N = len(target)
        rhs = -basis_functions.T@seis_forward2.vel_to_seis(basis_functions@x_guess, vec_adjoint=target, adjoint_on_residual=True)[2]/N
        rhs = rhs - np.concatenate( (self._prior_in_use.λ*self._prior_in_use.P@x_guess[:-1,:],cp.zeros((1,1),dtype=kgs.base_type_gpu)),axis=0)

        J_list = []
        import cupyx.scipy.linalg
        P = cupyx.scipy.linalg.block_diag(self._prior_in_use.λ*self._prior_in_use.P, cp.zeros((1,1),dtype=kgs.base_type_gpu))           
        J = cp.zeros((target.shape[0],basis_functions.shape[1]), kgs.base_type_gpu)
        for i in range(basis_functions.shape[1]): 
            #if i%100==0:
            #    print(i)
            _,diff,_ = seis_forward2.vel_to_seis(vec, vec_diff=basis_functions[:,i:i+1])
            J[:,i] = (diff[:,0])
        #print(time.time()-t)
        A = P+(J.T@J)/N
        #print(time.time()-t)
        del J
        res_inv = cp.linalg.solve(A,rhs)
        res_inv = res_inv[:,0]       
        
        res_inv = res_inv[:,None]        
        res2 = basis_functions@res_inv
        
        scales = np.linspace(-0.1,1.1,50)
        
        vals1 = [];        
        vals2 = [];
        vals3 = [];
        for scale in scales:
            #print(x_guess+scale*res_inv)
            cost_total, cost_prior, cost_residual = cost_and_gradient(x_guess+scale*res_inv, target, self.prior, basis_functions, compute_gradient=False)
            
            vals1.append(cp.asnumpy(cost_total))
            vals2.append(cp.asnumpy(cost_prior))
            vals3.append(cp.asnumpy(cost_residual))

        optimal_scale = scales[np.argmin(vals1)]
        
        res = velocity_guess.to_vector() + optimal_scale*res2
        result = copy.deepcopy(velocity_guess)
        result.from_vector( res )

        return result

    def seis_to_vel_lbfgs(self, seismogram, velocity_guess, maxiter=0):
        basis_functions = self._prior_in_use.basis_vectors
        x_guess = cp.asnumpy(cp.linalg.solve(cp.array(basis_functions.T@basis_functions), basis_functions.T@(velocity_guess.to_vector())))
        x_guess = x_guess.astype(dtype=kgs.base_type)
        target = seismogram.to_vector()

        
        def cost_and_gradient_func(x):
            xx = cp.from_dlpack(to_dlpack(x))[:,None]
            cost,gradient,cost_prior, cost_residual = cost_and_gradient(xx, target, self._prior_in_use, basis_functions, compute_gradient=True)         
            if cost_residual<self.seis_error_tolerance:
                cost = 0*cost
                gradient = 0*gradient
            cost = cost*self.scaling
            gradient = gradient*self.scaling
            return from_dlpack(cost.toDlpack()), from_dlpack(gradient[:,0].toDlpack())
            #torch.tensor(cp.asnumpy(cost),device='cuda'), torch.tensor(cp.asnumpy(gradient[:,0]),device='cuda')

        result = seis_numerics.bfgs(cost_and_gradient_func, torch.tensor(x_guess[:,0], device='cuda'), maxiter, self.lbfgs_tolerance_grad, self.lbfgs_tolerance_change)
    
        # Extract final result
        final_result = result.detach().cpu().numpy()

        result = copy.deepcopy(velocity_guess)
        if np.any(np.isnan(final_result)):
            # Rare failure - use initial guess   
            print('Failure! Reusing initial guess.')
            result.to_cupy();
        else:                
            result.from_vector( basis_functions@cp.array(final_result)[:,None] )
    
        return result


    def _infer_single(self,data):
        global true_vel
        assert self.prior.prepped
        self._prior_in_use = copy.deepcopy(self.prior)
        self._prior_in_use.adapt(data.velocity_guess)
        data.velocity_guess.to_cupy() 
        assert(len(self.iter_list)==1)
        for maxiter in self.iter_list:
            if maxiter==0:
                data.velocity_guess = self.seis_to_vel_gn(data.seismogram, data.velocity_guess)
            else:
                assert maxiter>0
                data.velocity_guess = self.seis_to_vel_lbfgs(data.seismogram, data.velocity_guess, maxiter=maxiter)

        data.velocity_guess.data = cp.asnumpy(data.velocity_guess.data)
        data.velocity_guess.min_vel = cp.asnumpy(data.velocity_guess.min_vel)
        return data

    def _train(self, train_data, validation_data):
        self.prior.prep()