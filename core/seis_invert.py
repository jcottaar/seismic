'''
Implements the core algorithms for full waveform inversion, in the form of a kaggle_support.Model.
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
import cupyx.scipy.linalg

def cost_and_gradient(x, target, prior, basis_functions, compute_gradient=False):
# Function to compute our overall cost function and its gradient.
# Cost function for velocity map x is: p(x) + ||s(x)-t||_2^2/N
# p(x) is the penalty function defined by the prior (defined by 'prior' input)
# s(x) is the seismogram that would be measured from x
# t is the target seismogram (defined by 'target' input)
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
    prior: seis_prior.Prior = field(init=True, default_factory = seis_prior.RowTotalVariation)  # Prior definition
    maxiter = 0 # Maximum number of iterations for LBFGS. If 0, uses Gauss-Newton instead.
    scaling = 1e15 # Scaling applied to cost function (helps in numerics)
    lbfgs_tolerance_grad = 1e-7 # Stopping criterion for LBFGS
    lbfgs_tolerance_change = 0.0 # Stopping criterion for LBFGS
    seis_error_tolerance = 0.0 # Stopping criterion for LBFGS (based on MSE of seismogram error)

    _prior_in_use = 0 # Used internally to do some internal computations in the prior

    def seis_to_vel_gn(self, seismogram, velocity_guess):
        # Does a Gauss-Newton pass, i.e. minimizes the cost function with approximations on the Hessian
        # Uses the current velocity_guess as linearization point
        # Note that this can only be done for a prior that makes an explicit precision matrix P       

        ## Construct right-hand side
        basis_functions = self._prior_in_use.basis_vectors
        x_guess = cp.linalg.solve(cp.array(basis_functions.T@basis_functions), basis_functions.T@(velocity_guess.to_vector()))
        x_guess = x_guess.astype(dtype=kgs.base_type)
        target = seismogram.to_vector()
        vec = velocity_guess.to_vector()
        N = len(target)
        rhs = -basis_functions.T@seis_forward2.vel_to_seis(basis_functions@x_guess, vec_adjoint=target, adjoint_on_residual=True)[2]/N
        rhs = rhs - np.concatenate( (self._prior_in_use.λ*self._prior_in_use.P@x_guess[:-1,:],cp.zeros((1,1),dtype=kgs.base_type_gpu)),axis=0)

        ## Construct P and J matrices
        # For P, we need to add an element for the min_vel degree of freedom
        P = cupyx.scipy.linalg.block_diag(self._prior_in_use.λ*self._prior_in_use.P, cp.zeros((1,1),dtype=kgs.base_type_gpu))    
        # Construct J column by column                   
        J = cp.empty((target.shape[0],basis_functions.shape[1]), kgs.base_type_gpu)
        for i in range(basis_functions.shape[1]): 
            _,diff,_ = seis_forward2.vel_to_seis(vec, vec_diff=basis_functions[:,i:i+1])
            J[:,i] = (diff[:,0])

        ## Solve the equation
        A = P+(J.T@J)/N
        del J
        res_inv = cp.linalg.solve(A,rhs)
        res_before_line_search = basis_functions@res_inv

        ## Crude line search
        scales = np.linspace(-0.1,1.1,50)        
        vals1 = [];        
        for scale in scales:
            cost_total, cost_prior, cost_residual = cost_and_gradient(x_guess+scale*res_inv, target, self.prior, basis_functions, compute_gradient=False)            
            vals1.append(cp.asnumpy(cost_total))
        optimal_scale = scales[np.argmin(vals1)]        
        res = velocity_guess.to_vector() + optimal_scale*res_before_line_search

        ## Store result
        result = copy.deepcopy(velocity_guess)
        result.from_vector( res )
        return result

    def seis_to_vel_lbfgs(self, seismogram, velocity_guess, maxiter=0):
        # Implements LBFGS, i.e. iteratively minimizing the cost function while constructing the Hessian from the iterates.
        # Starts from velocity_guess
        # Runs at most "maxiter"

        ## Collect stuff
        basis_functions = self._prior_in_use.basis_vectors
        target = seismogram.to_vector()

        ## Convert input to the basis in use        
        x_guess = cp.asnumpy(cp.linalg.solve(cp.array(basis_functions.T@basis_functions), basis_functions.T@(velocity_guess.to_vector())))
        x_guess = x_guess.astype(dtype=kgs.base_type)
        
        ## Define cost function
        def cost_and_gradient_func(x):
            # Torch to CuPy
            xx = cp.from_dlpack(to_dlpack(x))[:,None]
            # Compute cost and gradient
            cost,gradient,cost_prior, cost_residual = cost_and_gradient(xx, target, self._prior_in_use, basis_functions, compute_gradient=True)         
            # Stopping criterion
            if cost_residual<self.seis_error_tolerance:
                cost = 0*cost
                gradient = 0*gradient
            # Scaling for numerics
            cost = cost*self.scaling
            gradient = gradient*self.scaling
            # CuPy to Torch
            return from_dlpack(cost.toDlpack()), from_dlpack(gradient[:,0].toDlpack())

        ## Run LBFGS
        bfgs_result = seis_numerics.bfgs(cost_and_gradient_func, torch.tensor(x_guess[:,0], device='cuda'), maxiter, self.lbfgs_tolerance_grad, self.lbfgs_tolerance_change)
        bfgs_result = bfgs_result.detach().cpu().numpy()

        ## Store result
        result = copy.deepcopy(velocity_guess)
        if np.any(np.isnan(bfgs_result)):
            # Rare failure - use initial guess   
            print('Failure! Reusing initial guess.')
            result.to_cupy();
        else:                
            result.from_vector( basis_functions@cp.array(bfgs_result)[:,None] )    
        return result


    def _infer_single(self,data):
        ## Set up prior
        assert self.prior.prepped
        self._prior_in_use = copy.deepcopy(self.prior)
        self._prior_in_use.adapt(data.velocity_guess)
        data.velocity_guess.to_cupy() 

        ## Call the selected algorithm
        if self.maxiter==0:
            data.velocity_guess = self.seis_to_vel_gn(data.seismogram, data.velocity_guess)
        else:
            assert self.maxiter>0
            data.velocity_guess = self.seis_to_vel_lbfgs(data.seismogram, data.velocity_guess, maxiter=self.maxiter)

        ## Store results on CPU
        data.velocity_guess.data = cp.asnumpy(data.velocity_guess.data)
        data.velocity_guess.min_vel = cp.asnumpy(data.velocity_guess.min_vel)
        return data

    def _train(self, train_data, validation_data):
        # No real training, just do any preparations needed for the prior
        self.prior.prep()