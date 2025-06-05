import numpy as np
import cupy as cp
import kaggle_support as kgs
import seis_forward2
import seis_prior
import scipy
import copy
import time
from dataclasses import dataclass, field, fields
print('maxcor')

@kgs.profile_each_line
def cost_and_gradient(x, target, prior, basis_functions, compute_gradient=False):

    # Prior part
    if compute_gradient:
        cost_prior, gradient_prior = prior.compute_cost_and_gradient(x, compute_gradient=True)
    else:
        cost_prior = prior.compute_cost_and_gradient(x, compute_gradient=False)

    cp.cuda.Stream.null.synchronize()

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
        return cost_prior + cost_residual, gradient_prior + gradient_residual, cost_prior, cost_residual
    else:
        return cost_prior + cost_residual, cost_prior, cost_residual

true_vel = None
last_t = time.time()
def seis_to_vel(seismogram, velocity_guess, prior, scaling=1e10, maxiter=2000, method='BFGS'):
    basis_functions = prior.basis_functions()
    x_guess = cp.asnumpy(cp.linalg.solve(cp.array(basis_functions.T@basis_functions), basis_functions.T@(velocity_guess.to_vector())))
    x_guess = x_guess.astype(dtype=kgs.base_type)
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
        #global last_t
        #print('overhead: ', time.time()-last_t)
        xx = cp.array(x,dtype=kgs.base_type_gpu)[:,None]
        cost,gradient,cost_prior, cost_residual = cost_and_gradient(xx, target, prior, basis_functions, compute_gradient=True)
        if not true_vel is None:
            #print(cost, kgs.rms(basis_functions@cp.array(x[:,None])-true_vel.to_vector()))
            diagnostics['vel_error_per_fev'].append(cp.asnumpy(kgs.rms(basis_functions@xx-true_vel.to_vector())))
        diagnostics['seis_error_per_fev'].append(cp.asnumpy(cost_residual))
        diagnostics['prior_cost_per_fev'].append(cp.asnumpy(cost_prior))
        cost = cost*scaling
        gradient = gradient*scaling
        #last_t = time.time()
        return cp.asnumpy(cost), cp.asnumpy(gradient[:,0])

    #cost_func = lambda x: 
    #gradient_func = lambda x: 

    #res = scipy.optimize.minimize(cost_func, x_guess[:,0], method = 'L-BFGS-B', jac = gradient_func, options={'maxiter':maxiter})
    diagnostics = dict()
    diagnostics['vel_error_per_fev'] = []
    diagnostics['seis_error_per_fev'] = []
    diagnostics['prior_cost_per_fev'] = []
    if method=='BFGS':
        res = scipy.optimize.minimize(cost_and_gradient_func, x_guess[:,0], method = method, jac = True, options={'maxiter':maxiter})
    else:
        res = scipy.optimize.minimize(cost_and_gradient_func, x_guess[:,0], method = method, jac = True, options={'maxiter':maxiter, 'maxcor':1000})
    diagnostics['nit'] = res.nit
    diagnostics['nfev'] = res.nfev

    result = copy.deepcopy(velocity_guess)
    result.from_vector( basis_functions@cp.array(res.x)[:,None] )

    return result, diagnostics

def seis_to_vel_torch(seismogram, velocity_guess, prior, scaling=1e10, maxiter=2000, history_size=1000):
    basis_functions = prior.basis_functions()
    x_guess = cp.asnumpy(cp.linalg.solve(cp.array(basis_functions.T@basis_functions), basis_functions.T@(velocity_guess.to_vector())))
    x_guess = x_guess.astype(dtype=kgs.base_type)
    target = seismogram.to_vector()

    def cost_and_gradient_func(x):
        #global last_t
        #print('overhead: ', time.time()-last_t)
        xx = cp.array(x,dtype=kgs.base_type_gpu)[:,None]
        cost,gradient,cost_prior, cost_residual = cost_and_gradient(xx, target, prior, basis_functions, compute_gradient=True)
        if not true_vel is None:
            #print(cost, kgs.rms(basis_functions@cp.array(x[:,None])-true_vel.to_vector()))
            diagnostics['vel_error_per_fev'].append(cp.asnumpy(kgs.rms(basis_functions@xx-true_vel.to_vector())))
        diagnostics['seis_error_per_fev'].append(cp.asnumpy(cost_residual))
        diagnostics['prior_cost_per_fev'].append(cp.asnumpy(cost_prior))
        cost = cost*scaling
        gradient = gradient*scaling
        #last_t = time.time()
        return cp.asnumpy(cost), cp.asnumpy(gradient[:,0])

    #cost_func = lambda x: 
    #gradient_func = lambda x: 

    #res = scipy.optimize.minimize(cost_func, x_guess[:,0], method = 'L-BFGS-B', jac = gradient_func, options={'maxiter':maxiter})
    diagnostics = dict()
    diagnostics['vel_error_per_fev'] = []
    diagnostics['seis_error_per_fev'] = []
    diagnostics['prior_cost_per_fev'] = []

    # Torch LBFGS implementation
    import torch
    import numpy as np
    
    # Initialize the parameter to optimize
    # Assume x_guess is a NumPy array of shape (n,1) or similar
    x0 = np.asarray(x_guess[:,0], dtype=np.float64)
    param = torch.nn.Parameter(torch.from_numpy(x0))
    
    # Create the LBFGS optimizer
    optimizer = torch.optim.LBFGS(
        [param],
        lr=1.0,
        max_iter=maxiter,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=history_size,
        line_search_fn="strong_wolfe"
    )
    
    # Initialize diagnostics counters
    diagnostics['nfev'] = 0
    
    def closure():
        optimizer.zero_grad()
        # Convert the parameter to NumPy for cost_and_gradient_func
        x_np = param.detach().cpu().numpy()
        # Compute cost and gradient using the existing function
        cost, grad = cost_and_gradient_func(x_np)
        # Update function evaluation count
        diagnostics['nfev'] += 1
        # Convert gradient back to torch and assign to param.grad
        grad_torch = torch.from_numpy(grad.astype(np.float64)).to(param.dtype)
        param.grad = grad_torch
        # Return cost as a torch Tensor
        return torch.tensor(cost, dtype=param.dtype)
    
    # Perform optimization
    optimizer.step(closure)


    # Extract final result
    final_result = param.detach().cpu().numpy()

    result = copy.deepcopy(velocity_guess)
    result.from_vector( basis_functions@cp.array(final_result)[:,None] )

    return result, diagnostics

     

@dataclass
class InversionModel(kgs.Model):
    prior: seis_prior.Prior = field(init=True, default_factory = seis_prior.RowTotalVariation)
    maxiter = 2000
    history_size = 1000
    scaling = 1e15

    def _infer_single(self,data):
        global true_vel
        if data.is_train:
            data.velocity.load_to_memory()
            true_vel = data.velocity
        else:
            true_vel = None
        data.velocity_guess.data = cp.array(data.velocity_guess.data)
        data.velocity_guess.min_vel = cp.array(data.velocity_guess.min_vel)
        data.velocity_guess, diagnostics = seis_to_vel_torch(data.seismogram, data.velocity_guess, self.prior, scaling=self.scaling, maxiter=self.maxiter, history_size=self.history_size)
        data.velocity_guess.data = cp.asnumpy(data.velocity_guess.data)
        data.velocity_guess.min_vel = cp.asnumpy(data.velocity_guess.min_vel)

        data.diagnostics['seis_to_vel'] = diagnostics
        if data.is_train:
            data.velocity.unload()

        return data