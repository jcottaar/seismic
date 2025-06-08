import numpy as np
import cupy as cp
import kaggle_support as kgs
import seis_forward2
import seis_prior
import scipy
import copy
import time
from dataclasses import dataclass, field, fields
import matplotlib.pyplot as plt

profiling=False
last_t=time.time()
@kgs.profile_each_line
def cost_and_gradient(x, target, prior, basis_functions, compute_gradient=False):

    # Prior part
    if compute_gradient:
        cost_prior, gradient_prior = prior.compute_cost_and_gradient(x, compute_gradient=True)
    else:
        cost_prior = prior.compute_cost_and_gradient(x, compute_gradient=False)

    cp.cuda.Stream.null.synchronize()

    # Residual part
    t=time.time()
    vec = basis_functions@x
    if compute_gradient:
        s, _, s_adjoint = seis_forward2.vel_to_seis(vec, vec_adjoint=target, adjoint_on_residual=True)
    else:
        s, _, _ = seis_forward2.vel_to_seis(vec)
    if profiling:
        print(f'vel_to_seis time: {1e3*(time.time()-t):.2f}')

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
    history_size = 10000
    scaling = 1e15
    lbfgs_tolerance_grad = 1e-7
    maxiter= 2000    
    prec_matrix: object = field(init=True, default_factory = lambda:cp.eye(4901))

    prior_in_use = 0
    
    iter_list = 0

    show_convergence = False

    def seis_to_vel_gn(self, seismogram, velocity_guess, diagnostics, maxiter=0):
        basis_functions = self.prior.basis_vectors
        
        x_guess = cp.linalg.solve(cp.array(basis_functions.T@basis_functions), basis_functions.T@(velocity_guess.to_vector()))
        x_guess = x_guess.astype(dtype=kgs.base_type)
        target = seismogram.to_vector()
        vec = velocity_guess.to_vector()

        N = len(target)
        rhs = -basis_functions.T@seis_forward2.vel_to_seis(basis_functions@x_guess, vec_adjoint=target, adjoint_on_residual=True)[2]/N # basis_functions.T@J.T@(target-vel_guess)
        rhs = rhs - np.concatenate( (self.prior.λ*self.prior.P@x_guess[:-1,:],cp.zeros((1,1),dtype=kgs.base_type_gpu)),axis=0)
               
        def A(x):
            x = x[:,None]
            v1 = seis_forward2.vel_to_seis(vec, vec_diff=basis_functions@x, vec_adjoint = target, adjoint_on_diff=True)[2]
            res = np.concatenate( (self.prior.λ*self.prior.P@x[:-1,:],cp.zeros((1,1),dtype=kgs.base_type_gpu)),axis=0)+ basis_functions.T@v1/N
            res = res[:,0]
            return res

        def callback(x):            
            if self.show_convergence:
                xx = x[:,None]+x_guess
                cost,cost_prior, cost_residual = cost_and_gradient(xx, target, self.prior, basis_functions, compute_gradient=False)
                for ii in range(maxiter):
                    if not true_vel is None:
                        diagnostics['vel_error_per_fev'].append(cp.asnumpy(cp.mean(cp.abs((basis_functions@xx-true_vel.to_vector())))))
                    diagnostics['seis_error_per_fev'].append(cp.asnumpy(cost_residual))
                    diagnostics['total_cost_per_fev'].append(cp.asnumpy(cost))                    
                    diagnostics['x'].append(cp.asnumpy(basis_functions@xx))
            
        
        import cupyx.scipy.sparse.linalg
        AA=cupyx.scipy.sparse.linalg.LinearOperator( (self.prior.N,self.prior.N), A)        
        res_inv=cupyx.scipy.sparse.linalg.gmres(AA,rhs,maxiter=maxiter,restart=maxiter)[0]
        callback(0*res_inv)
        callback(res_inv)
        res_inv = res_inv[:,None]        
        res2 = basis_functions@res_inv
        res = velocity_guess.to_vector() + res2
        result = copy.deepcopy(velocity_guess)
        result.from_vector( res )

        return result, diagnostics

    def seis_to_vel_lbfgs(self, seismogram, velocity_guess, diagnostics, maxiter=0):
        basis_functions = self.prior.basis_vectors
        x_guess = cp.asnumpy(cp.linalg.solve(cp.array(basis_functions.T@basis_functions), basis_functions.T@(velocity_guess.to_vector())))
        x_guess = x_guess.astype(dtype=kgs.base_type)
        target = seismogram.to_vector()
    
        def cost_and_gradient_func(x):
            global last_t            
            start_t=time.time()
            xx = cp.array(x,dtype=kgs.base_type_gpu)[:,None]
            cost,gradient,cost_prior, cost_residual = cost_and_gradient(xx, target, self.prior, basis_functions, compute_gradient=True)
            if not true_vel is None:
                #print(cost, kgs.rms(basis_functions@cp.array(x[:,None])-true_vel.to_vector()))
                diagnostics['vel_error_per_fev'].append(cp.asnumpy(cp.mean(cp.abs(basis_functions@xx-true_vel.to_vector()))))
            diagnostics['seis_error_per_fev'].append(cp.asnumpy(cost_residual))
            diagnostics['total_cost_per_fev'].append(cp.asnumpy(cost))
            if self.show_convergence:
                diagnostics['x'].append(cp.asnumpy(basis_functions@xx))
            cost = cost*self.scaling
            gradient = gradient*self.scaling
            if profiling:               
                print(f'outside cost_and_gradient_func: {1e3*(start_t-last_t):.2f}')
                print(f'total iteration time: {1e3*(time.time()-last_t):.2f}')
                print('')
            last_t = time.time()
            return cp.asnumpy(cost), cp.asnumpy(gradient[:,0])

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
            tolerance_grad=self.lbfgs_tolerance_grad,
            tolerance_change=0.,
            history_size=self.history_size,
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


    def _infer_single(self,data):
        global true_vel
        assert self.prior.prepped
        if data.is_train:
            data.velocity.load_to_memory()
            true_vel = data.velocity
        else:
            true_vel = None
        diagnostics = dict()
        diagnostics['vel_error_per_fev'] = []
        diagnostics['seis_error_per_fev'] = []
        diagnostics['total_cost_per_fev'] = []
        diagnostics['x'] = []
        data.velocity_guess.to_cupy()       
        for maxiter in self.iter_list:
            if maxiter<0:
                data.velocity_guess, diagnostics = self.seis_to_vel_gn(data.seismogram, data.velocity_guess, diagnostics, maxiter=-maxiter)
            else:
                data.velocity_guess, diagnostics = self.seis_to_vel_lbfgs(data.seismogram, data.velocity_guess, diagnostics, maxiter=maxiter)
        # if self.show_convergence:
        #     x_by_it = []
        #     for x_interm in diagnostics['x'][::np.ceil(len(diagnostics['x'])/100).astype(int)]:
        #         x_by_it.append(np.reshape(x_interm[:-1,:],(70,70)) - cp.asnumpy(true_vel.data))
        #     x_by_it = np.stack(x_by_it)
        #     import seis_diagnostics
        #     seis_diagnostics.animate_3d_matrix(x_by_it)
        #     plt.pause(0.001)
            
        data.velocity_guess.data = cp.asnumpy(data.velocity_guess.data)
        data.velocity_guess.min_vel = cp.asnumpy(data.velocity_guess.min_vel)

        data.diagnostics['seis_to_vel'] = diagnostics
        if data.is_train:
            data.velocity.unload()

        return data

    def _train(self, train_data, validation_data):
        self.prior.prep()