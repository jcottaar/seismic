import numpy as np
import cupy as cp
import kaggle_support as kgs
from dataclasses import dataclass, field, fields
import matplotlib.pyplot as plt

@dataclass
class Prior(kgs.BaseClass):
    N: int = field(init=False, default=-1) # number of basis functions
    λ: float = field(init=False, default=1.)

    def _check_constraints(self):
        assert(self.N>0)
    

    def basis_functions(self):
        res = self._basis_functions()
        #assert type(res)==cp.ndarray
        assert res.dtype == kgs.base_type_gpu
        assert res.shape == (4901, self.N)
        return res

    def compute_cost_and_gradient(self, x, compute_gradient = False):
        assert x.shape == (self.N,1)

        cost, gradient = self._compute_cost_and_gradient(x, compute_gradient)        
        
        cost = self.λ * cost
        assert type(cost)==cp.ndarray
        assert cost.dtype == kgs.base_type_gpu
        assert cost.shape == ()
        if compute_gradient:
            gradient = self.λ * gradient
            assert type(gradient)==cp.ndarray
            assert gradient.dtype == kgs.base_type_gpu
            assert gradient.shape == (self.N,1)
            return cost, gradient
        else:
            return cost

@dataclass
class RowTotalVariation(Prior):
    epsilon: float = field(init=True, default=0.1)

    def __post_init__(self):
        # Mark the object as frozen after initialization        
        super().__post_init__()
        self.N = 71
        self.λ = 1e-8
    
    def _basis_functions(self):
        basis_vectors = []
        for i_row in range(70):
            mat = np.zeros((70,70),dtype=kgs.base_type)
            mat[i_row,:]=1.
            basis_vectors.append(np.concatenate((mat.flatten(), np.array([0]))))
        basis_vectors.append(np.concatenate((0*mat.flatten(), np.array([1]))))        
        basis_vectors = np.stack(basis_vectors)
        basis_vectors = basis_vectors.T
        basis_vectors=cp.array(basis_vectors, dtype=kgs.base_type_gpu)
        return basis_vectors

    def _compute_cost_and_gradient(self, x, compute_gradient):

        diff = cp.diff(x[:70],axis=0)
        cost_per_item = cp.sqrt(diff**2+self.epsilon**2)
        cost = cp.mean(cost_per_item)

        if compute_gradient:
            sign = diff / cost_per_item     
            gradient = cp.zeros_like(x)        
            gradient[0] = -sign[0]
            gradient[1:69] = sign[:-1] - sign[1:]            # sign[:-1] = sign[0..67], sign[1:] = sign[1..68]
            gradient[69] = sign[-1]
            gradient = gradient/69
        else:
            gradient = None

        return cost, gradient


@dataclass
class SquaredExponential(Prior):
    length_scale = np.log(32.4)
    noise = 0.1
    sigma = 183.4
    sigma_mean = 520
    sigma_slope = 31.4
    #svd_cutoff = 0.11
    K = 0
    P = 0
    

    compute_P = True
    transform = False

    def __post_init__(self):
        # Mark the object as frozen after initialization        
        super().__post_init__()
        self.N = 4901
        self.λ = 1e-8

    def _basis_functions(self):
        
        y = cp.arange(-35,35)[:,None]+cp.zeros( (1,70) )
        x = cp.arange(-35,35)[None,:]+cp.zeros( (70,1) )
        x = x.flatten()[:,None]
        y = y.flatten()[:,None]
        dist_matrix = cp.sqrt((x-x.T)**2+(y-y.T)**2)        
        K = (self.sigma**2)*cp.exp(-dist_matrix**2/(2*(self.length_scale)**2))        
        K = K+self.sigma_mean**2
        K = K+(self.sigma_slope**2)*(y@y.T)
        K = K+(self.noise**2)*cp.eye(4900)
        #print(self.sigma, self.sigma_mean, self.sigma_slope, self.noise)
        #K = (self.noise**2)*cp.eye(4900)
        K = K.astype(kgs.base_type_gpu)        
        self.K = K
        #plt.figure()
        U,s,_=cp.linalg.svd(self.K,compute_uv=True).get()
        #plt.semilogy(xx)
        #plt.title(xx[0]/xx[-1])
        plt.pause(0.001)
        if self.compute_P:
            self.P = cp.linalg.inv(K)

        #import cupyx.scipy.sparse
        #basis_vectors = cupyx.scipy.sparse.identity(4901, dtype=kgs.base_type_gpu)
        
            
        basis_vectors = cp.eye(4901, dtype=kgs.base_type_gpu)
        if transform:
            raise 'add min_vel'
            basis_vectors = U*cp.diag(cp.sqrt(s))
            self.K = cp.eye(4900)
            self.P = self.K
        return basis_vectors

    def _compute_cost_and_gradient(self, x, compute_gradient):

        cost = x[:-1,:].T@self.P@x[:-1,:]
        cost = cost[0,0]

        if compute_gradient:
            gradient = 2*cp.concatenate((self.P@x[:-1,:], cp.zeros((1,1),kgs.base_type_gpu)))
        else:
            gradient = None

        return cost, gradient