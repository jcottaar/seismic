'''
Implements priors for use in Bayesian full waveform inversion, as done in seis_invert.py
'''


import numpy as np
import cupy as cp
import kaggle_support as kgs
from dataclasses import dataclass, field, fields
import seis_numerics
import scipy.linalg
import cupyx.scipy.ndimage
import cupyx.scipy.linalg
import cupyx.scipy.sparse
import copy

@dataclass
class Prior(kgs.BaseClass):
    # Abstract prior class
    N: int = field(init=False, default=-1) # number of basis functions
    λ: float = field(init=False, default=1.) # prefactor applied to penalty function
    prepped: bool = field(init=False, default=False) # if the prior has been prepared by calling its 'prep' function
    basis_vectors = 0 # the actual basis functions

    def _check_constraints(self):
        assert(self.N>0)

    def prep(self):
        # Prepare the prior, implemented in subclass in _prep
        # Must at least compute the basis functions
        if not self.prepped:
            self._prep()
        self.prepped = True
        assert self.basis_vectors.dtype == kgs.base_type_gpu
        assert self.basis_vectors.shape == (4901, self.N)

    def adapt(self, velocity_guess_np):
        # Adapt to a specific velocity profile, implemented in subclass in _adapt
        assert self.prepped
        self._adapt(velocity_guess_np)
        assert self.basis_vectors.dtype == kgs.base_type_gpu
        assert self.basis_vectors.shape == (4901, self.N)

    def _adapt(self, velocity_guess_np):
        pass

    def compute_cost_and_gradient(self, x, compute_gradient = False):
        # Computes the unnormalized log likelihood (cost function) associated with this prior for a given velocity profile x, as well as its gradient.
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
    # A 1D total variation prior. Each row must be constant. 
    # The log likelihood/cost function is the mean of the the absolute values of the differences between the rows.
    # Absolute value is smoothed near zero with factor epsilon.
    epsilon: float = field(init=True, default=0.1)

    def __post_init__(self):
        # Mark the object as frozen after initialization        
        super().__post_init__()
        self.N = 71
        self.λ = 1e-8
    
    def _prep(self):
        basis_vectors = []
        for i_row in range(70):
            mat = np.zeros((70,70),dtype=kgs.base_type)
            mat[i_row,:]=1.
            basis_vectors.append(np.concatenate((mat.flatten(), np.array([0]))))
        basis_vectors.append(np.concatenate((0*mat.flatten(), np.array([1])))) # min_vel basis function       
        basis_vectors = np.stack(basis_vectors)
        basis_vectors = basis_vectors.T
        basis_vectors=cp.array(basis_vectors, dtype=kgs.base_type_gpu)
        self.basis_vectors = basis_vectors

    def _compute_cost_and_gradient(self, x, compute_gradient):

        diff = cp.diff(x[:70],axis=0)
        cost_per_item = cp.sqrt(diff**2+self.epsilon**2)
        cost = cp.mean(cost_per_item)

        if compute_gradient:
            # Thanks ChatGPT!
            sign = diff / cost_per_item     
            gradient = cp.zeros_like(x)        
            gradient[0] = -sign[0]
            gradient[1:69] = sign[:-1] - sign[1:]
            gradient[69] = sign[-1]
            gradient = gradient/69
        else:
            gradient = None

        return cost, gradient


@dataclass
class TotalVariation(Prior):
    # A 2D total variation prior.
    # The log likelihood/cost function is the mean of the the absolute values of the differences of each point with its neighbors.
    # Absolute value is smoothed near zero with factor epsilon.
    epsilon: float = field(init=True, default=0.1)
    cost_func = 0 # support for other cost function than absolute value - not used in this competition
    grad_cost_func = 0
    

    def __post_init__(self):
        # Mark the object as frozen after initialization        
        super().__post_init__()
        self.N = 4901
        self.λ = 1e-8

        self.cost_func = lambda x:cp.sqrt(x**2 + self.epsilon**2)
        self.grad_cost_func = lambda x:x/cp.sqrt(x**2 + self.epsilon**2)
    
    def _prep(self):
        basis_vectors=cupyx.scipy.sparse.csc_matrix(cp.eye(4901, dtype=kgs.base_type_gpu))
        self.basis_vectors = basis_vectors

    def _compute_cost_and_gradient(self, x, compute_gradient):
    
        # reshape input (excluding last element if used for something else)
        x_mat = cp.reshape(x[:-1], (70, 70))
        # forward differences
        diff1 = cp.diff(x_mat, axis=0)    # shape (69,70)
        diff2 = cp.diff(x_mat, axis=1)    # shape (70,69)
        # flatten and concatenate
        d1_flat = diff1.ravel()
        d2_flat = diff2.ravel()
        diff = cp.concatenate((d1_flat, d2_flat))  # shape (N_diff,)
    
        # cost per element and total cost
        #cost_per_item = cp.sqrt(diff**2 + self.epsilon**2)
        cost_per_item = self.cost_func(diff)
        cost = cp.mean(cost_per_item)
    
        if compute_gradient:
            # number of diff elements
            N = diff.size

            g1 = self.grad_cost_func(d1_flat)/N
            g2 = self.grad_cost_func(d2_flat)/N
            # reshape
            g1_mat = g1.reshape(diff1.shape)
            g2_mat = g2.reshape(diff2.shape)
    
            # backprop to x_mat
            grad_mat = cp.zeros_like(x_mat)
            # axis 0 diffs: + at i+1, - at i
            grad_mat[1:, :] += g1_mat
            grad_mat[:-1, :] -= g1_mat
            # axis 1 diffs: + at j+1, - at j
            grad_mat[:, 1:] += g2_mat
            grad_mat[:, :-1] -= g2_mat
    
            # flatten and append zero gradient for last element
            grad_vec = cp.concatenate((grad_mat.ravel(), cp.array([0], dtype=grad_mat.dtype)))
            gradient = grad_vec.reshape(x.shape)
        else:
            gradient = None
    
        return cost, gradient



@dataclass
class SquaredExponential(Prior):
    # Gaussian Process prior with squared exponential kernel.
    length_scale = np.log(32.4) # length scale for the SE kernel
    noise = 0.1 # noise level 
    sigma = 183.4 # sigma for the SE kernel
    sigma_mean = 520 # mean for a constant offset
    sigma_slope = 31.4 # mean for a slope over the depth
    
    transform = False # if transform = True, we take as basis function our SVD modes
    svd_cutoff = 0. # cutoff for the transform above, cutting out low-magnitude SVD modes

    P = 0 # precompute precision matrix

    def __post_init__(self):
        # Mark the object as frozen after initialization        
        super().__post_init__()
        self.N = 4901
        self.λ = 1e-8

    def _prep(self):

        # Compute covariance matrix
        y = cp.arange(-35,35)[:,None]+cp.zeros( (1,70) )
        x = cp.arange(-35,35)[None,:]+cp.zeros( (70,1) )
        x = x.flatten()[:,None]
        y = y.flatten()[:,None]
        dist_matrix = cp.sqrt((x-x.T)**2+(y-y.T)**2)        
        K = (self.sigma**2)*cp.exp(-dist_matrix**2/(2*(self.length_scale)**2))        
        K = K+self.sigma_mean**2
        K = K+(self.sigma_slope**2)*(y@y.T)
        K = K+(self.noise**2)*cp.eye(4900)        #
        K = K.astype(kgs.base_type_gpu)        

        # Compute precision matrix
        self.P = cp.linalg.inv(K)
                    
        # Pick SVD modes
        if self.transform:
            U,s,_=cp.linalg.svd(K,compute_uv=True)
            to_keep = s>self.svd_cutoff
            basis_vectors = (U[:,to_keep]@cp.diag(cp.sqrt(s[to_keep])))
            basis_vectors = cp.pad(basis_vectors, ((0, 1), (0, 1)), mode='constant', constant_values=0)
            basis_vectors[-1, -1] = 1. # add the min_vel degree of freedom
            self.P = cp.eye(basis_vectors.shape[1]-1)
        else:
            basis_vectors = cupyx.scipy.sparse.csc_matrix(cp.eye(4901, dtype=kgs.base_type_gpu))

        self.N = basis_vectors.shape[1]
        self.basis_vectors = basis_vectors

    def _compute_cost_and_gradient(self, x, compute_gradient):

        cost = x[:-1,:].T@self.P@x[:-1,:] # don't use the last value of x, corresponding to min_vel
        cost = cost[0,0]

        if compute_gradient:
            gradient = 2*cp.concatenate((self.P@x[:-1,:], cp.zeros((1,1),kgs.base_type_gpu)))
        else:
            gradient = None

        return cost, gradient

@dataclass
class RestrictFlatAreas(Prior):
    # Adds an additional restriction to a given prior:
    # Areas that are approximately flat in the given velocity profile must be exactly flat.
    underlying_prior = 0 # the actual prior
    diff_threshold = 1. # an area with differences smaller than this is considered flat

    def _prep(self):
        self.underlying_prior.prep()
        assert(cp.all(self.underlying_prior.basis_vectors==cp.eye(4901))) # other variations not supported
        self.basis_vectors = self.underlying_prior.basis_vectors
        self.N = 4901

    def _adapt(self, velocity_guess_np):
        # Find flat areas
        mat = velocity_guess_np.data
        labels,count = seis_numerics.label_thresholded_components(mat, self.diff_threshold, connectivity=4)
        labels = labels.ravel()

        # Construct associated basis
        basis_functions = np.zeros((4900,count+1), dtype=kgs.base_type)    
        for ind in range(4900):
            basis_functions[ind,labels[ind]]=1.
        basis_functions = cp.array(basis_functions)
        basis_functions = basis_functions[:,cp.sum(basis_functions,axis=0)>0]
        self.basis_vectors = cupyx.scipy.linalg.block_diag(basis_functions, cp.array([[1]], dtype=kgs.base_type_gpu)) # add min_vel
        self.basis_vectors = self.basis_vectors/cp.sum(self.basis_vectors,axis=0) # scale
        self.N = self.basis_vectors.shape[1]

    def _compute_cost_and_gradient(self, x, compute_gradient):
        underlying_x = self.basis_vectors@x        
        if compute_gradient:
            cost, underlying_gradient = self.underlying_prior.compute_cost_and_gradient(underlying_x, compute_gradient)
            gradient = self.basis_vectors.T@underlying_gradient
        else:
            cost = self.underlying_prior.compute_cost_and_gradient(underlying_x, compute_gradient)
            gradient = None
        return cost, gradient
        
        