import numpy as np
import cupy as cp
import kaggle_support as kgs
from dataclasses import dataclass, field, fields

@dataclass
class Prior(kgs.BaseClass):
    N: int = field(init=True, default=-1) # number of basis functions
    λ: float = field(init=True, default=1.)

    def _check_constraints(self):
        assert(self.N>0)
    

    def basis_functions(self):
        res = self._basis_functions()
        assert type(res)==cp.ndarray
        assert res.dtype == kgs.base_type_gpu
        assert res.shape == (4901, self.N)
        return res

    def compute_cost_and_gradient(self, x, compute_gradient = None):
        assert x.shape == (self.N,1)
        assert compute_gradient is None or compute_gradient.shape == (self.N,1)

        cost, gradient = self._compute_cost_and_gradient(x, compute_gradient)        
        
        cost = self.λ * cost
        assert type(cost)==cp.ndarray
        assert cost.dtype == kgs.base_type_gpu
        assert cost.shape == ()
        if not compute_gradient is None:
            gradient = self.λ * gradient
            assert type(gradient)==cp.ndarray
            assert gradient.dtype == kgs.base_type_gpu
            assert gradient.shape == ()

        return cost,gradient

@dataclass
class RowTotalVariation(Prior):
    epsilon: float = field(init=True, default=1.)

    def __post_init__(self):
        # Mark the object as frozen after initialization        
        super().__post_init__()
        self.N = 71
    
    def _basis_functions(self):
        basis_vectors = []
        for i_row in range(70):
            mat = np.zeros((70,70))
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
        cost = cp.sum(cost_per_item)

        if not compute_gradient is None:
            diff_grad = cp.diff(compute_gradient[:70],axis=0)
            cost_per_item_grad = diff_grad*diff/cost_per_item
            gradient = cp.sum(cost_per_item_grad)
        else:
            gradient = None

        return cost, gradient
