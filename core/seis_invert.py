import numpy as np
import cupy as cp
import kaggle_support as kgs
import seis_forward2
import seis_prior
import seis_numerics
import scipy
import copy
import time
from dataclasses import dataclass, field, fields
import matplotlib.pyplot as plt
import torch
from torch.utils.dlpack import to_dlpack, from_dlpack

profiling=False
last_t=time.time()
#@kgs.profile_each_line
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
    seis_error_tolerance = 0.0 # MSE, only for BFGS2
    maxiter= 2000    
    prec_matrix: object = field(init=True, default_factory = lambda:cp.eye(4901))

    prior_in_use = 0
    
    iter_list = 0
    lambda_list: object = field(init=True, default_factory=list)

    show_convergence = False
    use_new_bfgs = True

    _start_time = 0

    def seis_to_vel_gn(self, seismogram, velocity_guess, diagnostics, maxiter=0):
        basis_functions = self.prior_in_use.basis_vectors
        
        x_guess = cp.linalg.solve(cp.array(basis_functions.T@basis_functions), basis_functions.T@(velocity_guess.to_vector()))
        x_guess = x_guess.astype(dtype=kgs.base_type)
        target = seismogram.to_vector()
        vec = velocity_guess.to_vector()

        N = len(target)
        rhs = -basis_functions.T@seis_forward2.vel_to_seis(basis_functions@x_guess, vec_adjoint=target, adjoint_on_residual=True)[2]/N # basis_functions.T@J.T@(target-vel_guess)
        rhs = rhs - np.concatenate( (self.prior_in_use.λ*self.prior_in_use.P@x_guess[:-1,:],cp.zeros((1,1),dtype=kgs.base_type_gpu)),axis=0)

        def callback(x):            
            if self.show_convergence:
                xx = x[:,None]+x_guess
                cost,cost_prior, cost_residual = cost_and_gradient(xx, target, self.prior, basis_functions, compute_gradient=False)
                if maxiter==0:
                    mm = 4901//2
                else:
                    mm = maxiter
                for ii in range(mm):
                    if not true_vel is None:
                        diagnostics['vel_error_per_fev'].append(cp.asnumpy(cp.mean(cp.abs((basis_functions@xx-true_vel.to_vector())))))
                    diagnostics['seis_error_per_fev'].append(cp.asnumpy(cost_residual))
                    diagnostics['total_cost_per_fev'].append(cp.asnumpy(cost))       
                    diagnostics['time_per_fev'].append(time.time()-self._start_time)
                    diagnostics['x'].append(cp.asnumpy(basis_functions@xx))

        if maxiter==0: # exact solve
            J_list = []
            import cupyx.scipy.linalg
            P = cupyx.scipy.linalg.block_diag(self.prior_in_use.λ*self.prior_in_use.P, cp.zeros((1,1),dtype=kgs.base_type_gpu))           
            J = cp.zeros((target.shape[0],basis_functions.shape[1]), kgs.base_type_gpu)
            t=time.time()
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
        else: #GMRES
               
            def A(x):
                x = x[:,None]
                v1 = seis_forward2.vel_to_seis(vec, vec_diff=basis_functions@x, vec_adjoint = target, adjoint_on_diff=True)[2]
                res = np.concatenate( (self.prior_in_use.λ*self.prior_in_use.P@x[:-1,:],cp.zeros((1,1),dtype=kgs.base_type_gpu)),axis=0)+ basis_functions.T@v1/N
                res = res[:,0]
                return res
                
            
            import cupyx.scipy.sparse.linalg
            AA=cupyx.scipy.sparse.linalg.LinearOperator( (self.prior_in_use.N,self.prior_in_use.N), A)        
            res_inv=cupyx.scipy.sparse.linalg.gmres(AA,rhs,maxiter=maxiter,restart=maxiter)[0]

        callback(0*res_inv)
        
        res_inv = res_inv[:,None]        
        res2 = basis_functions@res_inv
        
        scales = np.linspace(-0.1,1.1,50)

        if not true_vel is None and self.show_convergence:
            vals = [];        
            for scale in scales:
                res_here = velocity_guess.to_vector() + scale*res2
                vals.append(cp.asnumpy(kgs.rms(res_here-true_vel.to_vector())))            
            plt.figure()
            plt.semilogy(scales,vals)
            plt.legend(('Vel error'))
            plt.grid(True)
            plt.pause(0.001)

        
        vals1 = [];        
        vals2 = [];
        vals3 = [];
        for scale in scales:
            #print(x_guess+scale*res_inv)
            cost_total, cost_prior, cost_residual = cost_and_gradient(x_guess+scale*res_inv, target, self.prior, basis_functions, compute_gradient=False)
            
            vals1.append(cp.asnumpy(cost_total))
            vals2.append(cp.asnumpy(cost_prior))
            vals3.append(cp.asnumpy(cost_residual))
        if self.show_convergence:
            plt.figure()
            plt.semilogy(scales,vals1)
            plt.semilogy(scales,vals2)
            plt.semilogy(scales,vals3)        
            plt.legend(('Cost total', 'Prior', 'Residual'))
            plt.grid(True)
            plt.pause(0.001)

        optimal_scale = scales[np.argmin(vals1)]

        callback(optimal_scale*res_inv)
        
        res = velocity_guess.to_vector() + optimal_scale*res2
        result = copy.deepcopy(velocity_guess)
        result.from_vector( res )

        return result, diagnostics

    def seis_to_vel_lbfgs(self, seismogram, velocity_guess, diagnostics, maxiter=0):
        basis_functions = self.prior_in_use.basis_vectors
        x_guess = cp.asnumpy(cp.linalg.solve(cp.array(basis_functions.T@basis_functions), basis_functions.T@(velocity_guess.to_vector())))
        x_guess = x_guess.astype(dtype=kgs.base_type)
        target = seismogram.to_vector()
         
        def cost_and_gradient_func(x):
            global last_t            
            start_t=time.time()
            xx = cp.array(x,dtype=kgs.base_type_gpu)[:,None]
            cost,gradient,cost_prior, cost_residual = cost_and_gradient(xx, target, self.prior_in_use, basis_functions, compute_gradient=True)
            if not true_vel is None:
                #print(cost, kgs.rms(basis_functions@cp.array(x[:,None])-true_vel.to_vector()))
                diagnostics['vel_error_per_fev'].append(cp.asnumpy(cp.mean(cp.abs(basis_functions@xx-true_vel.to_vector()))))
            diagnostics['seis_error_per_fev'].append(cp.asnumpy(cost_residual))
            diagnostics['total_cost_per_fev'].append(cp.asnumpy(cost))
            diagnostics['time_per_fev'].append(time.time()-self._start_time)
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

    def seis_to_vel_lbfgs2(self, seismogram, velocity_guess, diagnostics, maxiter=0):
        basis_functions = self.prior_in_use.basis_vectors
        x_guess = cp.asnumpy(cp.linalg.solve(cp.array(basis_functions.T@basis_functions), basis_functions.T@(velocity_guess.to_vector())))
        x_guess = x_guess.astype(dtype=kgs.base_type)
        target = seismogram.to_vector()

        
        def cost_and_gradient_func(x):
            global last_t            
            start_t=time.time()
            xx = cp.from_dlpack(to_dlpack(x))[:,None]
            cost,gradient,cost_prior, cost_residual = cost_and_gradient(xx, target, self.prior_in_use, basis_functions, compute_gradient=True)           
            if self.show_convergence:
                if not true_vel is None:
                    #print(cost, kgs.rms(basis_functions@cp.array(x[:,None])-true_vel.to_vector()))
                    diagnostics['vel_error_per_fev'].append(cp.asnumpy(cp.mean(cp.abs(basis_functions@xx-true_vel.to_vector()))))
                diagnostics['seis_error_per_fev'].append(cp.asnumpy(cost_residual))
                diagnostics['total_cost_per_fev'].append(cp.asnumpy(cost))
                diagnostics['time_per_fev'].append(time.time()-self._start_time)            
                diagnostics['x'].append(cp.asnumpy(basis_functions@xx))
            if cost_residual<self.seis_error_tolerance:
                cost = 0*cost
                gradient = 0*gradient
            cost = cost*self.scaling
            gradient = gradient*self.scaling
            if profiling:               
                print(f'outside cost_and_gradient_func: {1e3*(start_t-last_t):.2f}')
                print(f'total iteration time: {1e3*(time.time()-last_t):.2f}')
                print('')
            last_t = time.time()
            return from_dlpack(cost.toDlpack()), from_dlpack(gradient[:,0].toDlpack())
            #torch.tensor(cp.asnumpy(cost),device='cuda'), torch.tensor(cp.asnumpy(gradient[:,0]),device='cuda')

        result = seis_numerics.bfgs(cost_and_gradient_func, torch.tensor(x_guess[:,0], device='cuda'), maxiter, self.lbfgs_tolerance_grad)
    
        # Extract final result
        final_result = result.detach().cpu().numpy()

        result = copy.deepcopy(velocity_guess)
        if np.any(np.isnan(final_result)):
            # Rare failure - use initial guess   
            print('Failure! Reusing initial guess.')
            result.to_cupy();
        else:                
            result.from_vector( basis_functions@cp.array(final_result)[:,None] )
    
        return result, diagnostics


    def _infer_single(self,data):
        global true_vel
        assert self.prior.prepped
        self.prior_in_use = copy.deepcopy(self.prior)
        self.prior_in_use.adapt(data.velocity_guess)
        if data.is_train:
            data.velocity.load_to_memory()
            true_vel = data.velocity
        else:
            true_vel = data.velocity_guess
        diagnostics = dict()
        diagnostics['time_per_fev'] = []
        diagnostics['vel_error_per_fev'] = []
        diagnostics['seis_error_per_fev'] = []
        diagnostics['total_cost_per_fev'] = []
        diagnostics['x'] = []
        data.velocity_guess.to_cupy()       
        self._start_time = time.time()
        for imi, maxiter in enumerate(self.iter_list):
            if len(self.lambda_list)>imi:
                self.prior_in_use.λ = self.lambda_list[imi]
            if maxiter<=0:
                data.velocity_guess, diagnostics = self.seis_to_vel_gn(data.seismogram, data.velocity_guess, diagnostics, maxiter=-maxiter)
            else:
                if self.use_new_bfgs:
                    data.velocity_guess, diagnostics = self.seis_to_vel_lbfgs2(data.seismogram, data.velocity_guess, diagnostics, maxiter=maxiter)
                else:
                    data.velocity_guess, diagnostics = self.seis_to_vel_lbfgs(data.seismogram, data.velocity_guess, diagnostics, maxiter=maxiter)

        if self.show_convergence:
           # x_by_it = []
           # for x_interm in diagnostics['x'][::np.ceil(len(diagnostics['x'])/100).astype(int)]:
           #     x_by_it.append(np.reshape(x_interm[:-1,:],(70,70)) - cp.asnumpy(true_vel.data))
           # x_by_it = np.stack(x_by_it)
            print('MAE update: ', np.mean(np.abs(diagnostics['x'][0]-diagnostics['x'][-1])))
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