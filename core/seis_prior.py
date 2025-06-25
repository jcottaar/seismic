import numpy as np
import cupy as cp
import kaggle_support as kgs
from dataclasses import dataclass, field, fields
import matplotlib.pyplot as plt
import seis_numerics
import scipy.linalg
import cupyx.scipy.ndimage
import cupyx.scipy.linalg
import copy

@dataclass
class Prior(kgs.BaseClass):
    N: int = field(init=False, default=-1) # number of basis functions
    λ: float = field(init=False, default=1.)
    prepped: bool = field(init=False, default=False)

    basis_vectors = 0

    def _check_constraints(self):
        assert(self.N>0)

    def prep(self):
        if not self.prepped:
            self._prep()
        self.prepped = True
        assert self.basis_vectors.dtype == kgs.base_type_gpu
        assert self.basis_vectors.shape == (4901, self.N)

    def adapt(self, velocity_guess_np):
        assert self.prepped
        self._adapt(velocity_guess_np)
        assert self.basis_vectors.dtype == kgs.base_type_gpu
        assert self.basis_vectors.shape == (4901, self.N)

    def _adapt(self, velocity_guess_np):
        pass

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
    
    def _prep(self):
        basis_vectors = []
        for i_row in range(70):
            mat = np.zeros((70,70),dtype=kgs.base_type)
            mat[i_row,:]=1.
            basis_vectors.append(np.concatenate((mat.flatten(), np.array([0]))))
        basis_vectors.append(np.concatenate((0*mat.flatten(), np.array([1]))))        
        basis_vectors = np.stack(basis_vectors)
        basis_vectors = basis_vectors.T
        basis_vectors=cp.array(basis_vectors, dtype=kgs.base_type_gpu)
        self.basis_vectors = basis_vectors

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
class TotalVariation(Prior):
    epsilon: float = field(init=True, default=0.1)

    def __post_init__(self):
        # Mark the object as frozen after initialization        
        super().__post_init__()
        self.N = 4901
        self.λ = 1e-8
    
    def _prep(self):
        basis_vectors=cp.eye(4901, dtype=kgs.base_type_gpu)
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
        cost_per_item = cp.sqrt(diff**2 + self.epsilon**2)
        cost = cp.mean(cost_per_item)
    
        if compute_gradient:
            # number of diff elements
            N = diff.size
            # split cost_per_item back to match diff1 and diff2
            c1 = cost_per_item[:d1_flat.size]
            c2 = cost_per_item[d1_flat.size:]
            # gradient w.r.t. diff elements (chain through sqrt and mean)
            g1 = (d1_flat / c1) / N
            g2 = (d2_flat / c2) / N
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
    length_scale = np.log(32.4)
    noise = 0.1
    sigma = 183.4
    sigma_mean = 520
    sigma_slope = 31.4
    svd_cutoff = 0.

    transform = False

    #K = 0
    P = 0
    basis_vectors=0
    use_full = False

    def __post_init__(self):
        # Mark the object as frozen after initialization        
        super().__post_init__()
        self.N = 4901
        self.λ = 1e-8

    def _prep(self):
        
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
        #self.K = K
        #plt.figure()
        
        #plt.semilogy(xx)
        #plt.title(xx[0]/xx[-1])
        #plt.pause(0.001)
        self.P = cp.linalg.inv(K)

        #import cupyx.scipy.sparse
        #basis_vectors = cupyx.scipy.sparse.identity(4901, dtype=kgs.base_type_gpu)
        
            
        basis_vectors = cp.eye(4901, dtype=kgs.base_type_gpu)
        if self.transform:
            U,s,_=cp.linalg.svd(K,compute_uv=True)
            #plt.figure();plt.semilogy(s.get());plt.grid(True);plt.pause(0.001)
            to_keep = s>self.svd_cutoff
            basis_vectors = (U[:,to_keep]@cp.diag(cp.sqrt(s[to_keep])))
            basis_vectors = cp.pad(basis_vectors, ((0, 1), (0, 1)), mode='constant', constant_values=0)
            basis_vectors[-1, -1] = 1.
            #self.K = cp.eye(self.basis_vectors.shape[1]-1)
            self.P = cp.eye(basis_vectors.shape[1]-1)

        self.N = basis_vectors.shape[1]
        self.basis_vectors = basis_vectors

    def _compute_cost_and_gradient(self, x, compute_gradient):

        assert not self.use_full

        if self.use_full:
            cost = x.T@self.P@x
        else:            
            cost = x[:-1,:].T@self.P@x[:-1,:]
        cost = cost[0,0]

        if compute_gradient:
            if self.use_full:
                gradient = 2*cp.concatenate((self.P@x[:,:]))[:,None]
            else:
                gradient = 2*cp.concatenate((self.P@x[:-1,:], cp.zeros((1,1),kgs.base_type_gpu)))
        else:
            gradient = None

        return cost, gradient

@dataclass
class RestrictFlatAreas(Prior):
    underlying_prior = 0
    diff_threshold1 = 1.
    diff_threshold2 = 30.

    def _prep(self):

        self.underlying_prior.prep()
        assert(cp.all(self.underlying_prior.basis_vectors==cp.eye(4901)))
        self.basis_vectors = self.underlying_prior.basis_vectors
        self.N = 4901


    def _adapt(self, velocity_guess_np):
        mat = velocity_guess_np.data
        labels,count = seis_numerics.label_thresholded_components(mat, self.diff_threshold1, connectivity=4)

        mat = cp.array(mat)
        maxx = cupyx.scipy.ndimage.maximum_filter(mat,3)
        minn = -cupyx.scipy.ndimage.maximum_filter(-mat,3)
        diff = cp.maximum(cp.abs(mat-maxx), cp.abs(mat-minn))
        is_diff = (diff>self.diff_threshold2)
        diff_inds = cp.argwhere(cp.ravel(is_diff)).get()
    
    
        labels = labels.ravel()
        for i_ind,ind in enumerate(diff_inds):
            labels[ind] = i_ind+count+1

        plot_list=np.zeros(4900)
        #labels_mixed = copy.deepcopy(labels)
        #np.random.default_rng(seed=0).shuffle(labels_mixed)
        basis_functions = np.zeros((4900,count+len(diff_inds)+1), dtype=kgs.base_type)    
        for ind in range(4900):
            basis_functions[ind,labels[ind]]=1.
            plot_list[ind] = labels[ind]
        basis_functions = cp.array(basis_functions)
        basis_functions = basis_functions[:,cp.sum(basis_functions,axis=0)>0]

        self.basis_vectors = cupyx.scipy.linalg.block_diag(basis_functions, cp.array([[1]], dtype=kgs.base_type_gpu))
        self.basis_vectors = self.basis_vectors/cp.sum(self.basis_vectors,axis=0)
        self.N = self.basis_vectors.shape[1]
        print(self.N)
        plt.figure();plt.imshow(np.reshape(plot_list,(70,70)));plt.pause(0.001)

    def _compute_cost_and_gradient(self, x, compute_gradient):
        underlying_x = self.basis_vectors@x        
        if compute_gradient:
            cost, underlying_gradient = self.underlying_prior.compute_cost_and_gradient(underlying_x, compute_gradient)
            gradient = self.basis_vectors.T@underlying_gradient
        else:
            cost = self.underlying_prior.compute_cost_and_gradient(underlying_x, compute_gradient)
            gradient = None
        return cost, gradient
        
        