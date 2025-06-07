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
     

@dataclass
class InversionModel(kgs.Model):
    prior: seis_prior.Prior = field(init=True, default_factory = seis_prior.RowTotalVariation)
    maxiter = 2000
    history_size = 1000
    scaling = 1e15
    prec_matrix: object = field(init=True, default_factory = lambda:cp.eye(4901))

    do_gn = False

    show_convergence = False

    def seis_to_vel_gn(self, seismogram, velocity_guess):
        basis_functions = self.prior.basis_functions()
        if not cp.all(self.prec_matrix == cp.eye(4901)):
            basis_functions = self.prec_matrix@basis_functions
            self.prior.P = self.prec_matrix[:-1,:-1].T@self.prior.P@self.prec_matrix[:-1,:-1]
            print(self.prior.P.shape)
        x_guess = cp.linalg.solve(cp.array(basis_functions.T@basis_functions), basis_functions.T@(velocity_guess.to_vector()))
        x_guess = x_guess.astype(dtype=kgs.base_type)
        target = seismogram.to_vector()
        vec = velocity_guess.to_vector()

        rhs = -basis_functions.T@seis_forward2.vel_to_seis(basis_functions@x_guess, vec_adjoint=target, adjoint_on_residual=True)[2] # basis_functions.T@J.T@(target-vel_guess)
        rhs = rhs - np.concatenate( (self.prior.λ*self.prior.P@x_guess[:-1,:],cp.zeros((1,1),dtype=kgs.base_type_gpu)),axis=0)
        print(kgs.rms(rhs))
        print(kgs.rms(np.concatenate( (self.prior.λ*self.prior.P@x_guess[:-1,:],cp.zeros((1,1),dtype=kgs.base_type_gpu)),axis=0)))
        print('gn', kgs.rms(seismogram.to_vector()- seis_forward2.vel_to_seis(velocity_guess.to_vector())[0]))
        print(kgs.rms(-basis_functions.T@seis_forward2.vel_to_seis(vec, vec_adjoint=target, adjoint_on_residual=True)[2]))
        
        def A(x):
            x = x[:,None]
            v1 = seis_forward2.vel_to_seis(vec, vec_diff=basis_functions@x)[1]
            v2 = seis_forward2.vel_to_seis(vec, vec_adjoint=v1)[2]
            res = np.concatenate( (self.prior.λ*self.prior.P@x[:-1,:],cp.zeros((1,1),dtype=kgs.base_type_gpu)),axis=0)+ basis_functions.T@v2
            res = res[:,0]
            print(res.shape)
            return res

        import seis_forward
        print(basis_functions.shape)
        seis_forward.vel_to_seis_J_file(velocity_guess, kgs.temp_dir + '/Jint', n_split=30, rhs=basis_functions.T)
        J = seis_forward.vel_to_seis_J_load_file(kgs.temp_dir + '/Jint_64_', to_cpu=True, n_rhs=basis_functions.shape[1])        

        # test_vec = np.random.default_rng(seed=0).normal(0,1,(x_guess.shape[0],1))
        # rres1 = cp.array(J@test_vec)
        # rres2 = seis_forward2.vel_to_seis(vec, vec_diff = basis_functions@cp.array(test_vec))[1]
        # print(kgs.rms(rres1-rres2), kgs.rms(rres1))
        
        print(J.shape)
        A_mat = cp.pad(self.prior.λ*self.prior.P, ((0, 1), (0, 1)), mode='constant', constant_values=0) + cp.array(J.T@J)
        res_inv = cp.linalg.solve(A_mat, rhs)
        print('res_inv', kgs.rms(res_inv))
        print(res_inv.shape)
        # Numerically invert A
        # import cupyx.scipy.sparse.linalg
        # AA=cupyx.scipy.sparse.linalg.LinearOperator( (self.prior.N,self.prior.N), A)
        # print(rhs.shape)
        # res_inv=cupyx.scipy.sparse.linalg.gmres(AA,rhs,maxiter=1,restart=1)[0]
        # res_inv = res_inv[:,None]


        
        res2 = basis_functions@res_inv
       
        vals = [];vals2 = [];
        scales = np.linspace(-0.05,1.05,100)
        seis_offset = seis_forward2.vel_to_seis(vec)[0]
        Jres2 = cp.array(J@cp.asnumpy(res_inv))
        #scales = np.linspace(-0.01,0.01,100)
        for scale in scales:
            res_here = velocity_guess.to_vector() + scale*res2
            #vals.append(cp.asnumpy(kgs.rms(res_here-true_vel.to_vector())))
            vals.append(cp.asnumpy(kgs.rms(seismogram.to_vector() - seis_offset - scale*Jres2)))
            vals2.append(cp.asnumpy(kgs.rms(seismogram.to_vector() - seis_forward2.vel_to_seis(velocity_guess.to_vector() + scale*res2)[0])))
        plt.figure()
        plt.semilogy(scales,vals)
        plt.semilogy(scales,vals2)
        plt.legend(('Linearized', 'Actual'))
        plt.grid(True)
        plt.pause(0.001)

        vals = [];        
        for scale in scales:
            res_here = velocity_guess.to_vector() + scale*res2
            vals.append(cp.asnumpy(kgs.rms(res_here-true_vel.to_vector())))            
        plt.figure()
        plt.semilogy(scales,vals)
        plt.legend(('Vel error'))
        plt.grid(True)
        plt.pause(0.001)

        vals = [];        
        for scale in scales:
            #print(x_guess+scale*res_inv)
            vals.append(cp.asnumpy(cost_and_gradient(x_guess+scale*res_inv, target, self.prior, basis_functions, compute_gradient=False)[0]))
        #print(vals)
        plt.figure()
        plt.semilogy(scales,vals)
        plt.legend(('Cost'))
        plt.grid(True)
        plt.pause(0.001)

        scale_use = scales[np.argmin(vals)]

        res = velocity_guess.to_vector() + scale_use*res2

        result = copy.deepcopy(velocity_guess)
        result.from_vector( res )

            

        diagnostics = dict()
        diagnostics['vel_error_per_fev'] = []
        diagnostics['seis_error_per_fev'] = []
        diagnostics['prior_cost_per_fev'] = []
        if not true_vel is None:
            diagnostics['vel_error_per_fev'].append(cp.asnumpy(kgs.rms(res-true_vel.to_vector())))
        cost,cost_prior,cost_residual = cost_and_gradient(x_guess+res_inv, target, self.prior, basis_functions, compute_gradient=False)
        diagnostics['seis_error_per_fev'].append(cp.asnumpy(cost_residual))
        diagnostics['prior_cost_per_fev'].append(cp.asnumpy(cost_prior))
        
    
        return result, diagnostics

    def seis_to_vel_torch(self, seismogram, velocity_guess):
        basis_functions = self.prior.basis_functions()
        if not cp.all(self.prec_matrix == cp.eye(4901)):
            basis_functions = self.prec_matrix@basis_functions
            self.prior.P = self.prec_matrix[:-1,:-1].T@self.prior.P@self.prec_matrix[:-1,:-1]
            print(self.prior.P.shape)
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
                diagnostics['vel_error_per_fev'].append(cp.asnumpy(kgs.rms(basis_functions@xx-true_vel.to_vector())))
            diagnostics['seis_error_per_fev'].append(cp.asnumpy(cost_residual))
            diagnostics['prior_cost_per_fev'].append(cp.asnumpy(cost_prior))
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
    
        #cost_func = lambda x: 
        #gradient_func = lambda x: 
    
        #res = scipy.optimize.minimize(cost_func, x_guess[:,0], method = 'L-BFGS-B', jac = gradient_func, options={'maxiter':maxiter})
        diagnostics = dict()
        diagnostics['vel_error_per_fev'] = []
        diagnostics['seis_error_per_fev'] = []
        diagnostics['prior_cost_per_fev'] = []
        diagnostics['x'] = []
    
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
            max_iter=self.maxiter,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
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
        if data.is_train:
            data.velocity.load_to_memory()
            true_vel = data.velocity
        else:
            true_vel = None
        data.velocity_guess.data = cp.array(data.velocity_guess.data)
        data.velocity_guess.min_vel = cp.array(data.velocity_guess.min_vel)
        if self.do_gn:            
            for ii in range(10):
                plt.figure()
                plt.imshow(cp.asnumpy(data.velocity_guess.data - true_vel.data))
                plt.colorbar()
                plt.pause(0.001)
                data.velocity_guess, diagnostics = self.seis_to_vel_gn(data.seismogram, data.velocity_guess)
        else:            
            data.velocity_guess, diagnostics = self.seis_to_vel_torch(data.seismogram, data.velocity_guess)
            if self.show_convergence:
                x_by_it = []
                for x_interm in diagnostics['x'][::np.ceil(len(diagnostics['x'])/100).astype(int)]:
                    x_by_it.append(np.reshape(x_interm[:-1,:],(70,70)) - cp.asnumpy(true_vel.data))
                x_by_it = np.stack(x_by_it)
                import seis_diagnostics
                seis_diagnostics.animate_3d_matrix(x_by_it)
                plt.pause(0.001)
                #mat = np.random.default_rng(seed=0).normal(0,1,size=((100,100,100)))
                #seis_diagnostics.animate_3d_matrix(mat)
                #raise 'stop'
            #data.velocity_guess, diagnostics = self.seis_to_vel_gn(data.seismogram, data.velocity_guess)
            
        data.velocity_guess.data = cp.asnumpy(data.velocity_guess.data)
        data.velocity_guess.min_vel = cp.asnumpy(data.velocity_guess.min_vel)

        data.diagnostics['seis_to_vel'] = diagnostics
        if data.is_train:
            data.velocity.unload()

        return data