import cupy as cp
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import lil_matrix
import copy
import kaggle_support as kgs
from torch.utils.dlpack import to_dlpack, from_dlpack

def closest_neighbor_values(A):
    """
    For each element in A, find among its up/down/left/right neighbors
    the one whose value is closest to A[i,j], and store that neighbor's
    value in the output array B.
    """
    m, n = A.shape
    
    # Prepare shifted versions of A; out-of-bounds positions get a large dummy difference
    # so they will never be chosen.
    # shift up
    A_up = cp.empty_like(A);   A_up[1:,:] = A[:-1,:];   A_up[0,:] = cp.inf
    # shift down
    A_dn = cp.empty_like(A);   A_dn[:-1,:] = A[1:,:];    A_dn[-1,:] = cp.inf
    # shift left
    A_lf = cp.empty_like(A);   A_lf[:,1:] = A[:,:-1];   A_lf[:,0] = cp.inf
    # shift right
    A_rt = cp.empty_like(A);   A_rt[:,:-1] = A[:,1:];    A_rt[:,-1] = cp.inf

    # Stack all neighbor values and compute absolute differences:
    # shape (m, n, 4)
    neigh_vals = cp.stack([A_up, A_dn, A_lf, A_rt], axis=-1)
    diffs      = cp.abs(neigh_vals - A[..., cp.newaxis])

    # Find index of the closest neighbor along the last axis, then pick its value
    idx_closest = cp.argmin(diffs, axis=-1)
    B = cp.take_along_axis(neigh_vals, idx_closest[..., cp.newaxis], axis=-1)[..., 0]

    return B
    
def unpad_edge_padded_gradient(v_adjoint: cp.ndarray, nbc: int) -> cp.ndarray:
    """
    Given a gradient array `v_adjoint` that was computed on an edge-padded field (mode='edge'),
    fold (sum) all contributions from the padded border back into the corresponding boundary cells
    of the unpadded region, then return the unpadded result.

    Parameters
    ----------
    v_adjoint : cp.ndarray
        A 2D CuPy array of shape (H, W), where H = N + 2*nbc and W = M + 2*nbc. It represents
        the gradient w.r.t. a field that was originally of shape (N, M) and then padded by `nbc`
        on each side with mode='edge'.
    nbc : int
        Number of boundary‐padding cells on each side.

    Returns
    -------
    cp.ndarray
        A 2D CuPy array of shape (N, M), which is the gradient folded back into the original
        (unpadded) domain.
    """
    # Full padded gradient array
    W_full = v_adjoint
    H, W = W_full.shape

    # Dimensions of the unpadded region
    N = H - 2 * nbc
    M = W - 2 * nbc

    # Extract the interior (unpadded) slice
    g = W_full[nbc:-nbc, nbc:-nbc].copy()  # shape = (N, M)

    # Top edge padding → fold into row 0 of `g`
    top_block = W_full[0:nbc, nbc:-nbc]        # shape = (nbc, M)
    g[0, :] += top_block.sum(axis=0)

    # Bottom edge padding → fold into row N-1 of `g`
    bottom_block = W_full[-nbc:, nbc:-nbc]      # shape = (nbc, M)
    g[-1, :] += bottom_block.sum(axis=0)

    # Left edge padding → fold into column 0 of `g`
    left_block = W_full[nbc:-nbc, 0:nbc]        # shape = (N, nbc)
    g[:, 0] += left_block.sum(axis=1)

    # Right edge padding → fold into column M-1 of `g`
    right_block = W_full[nbc:-nbc, -nbc:]       # shape = (N, nbc)
    g[:, -1] += right_block.sum(axis=1)

    # Four corner blocks → fold into the four corners of `g`
    # Top-left corner
    corner_tl = W_full[0:nbc, 0:nbc]            # shape = (nbc, nbc)
    g[0, 0] += corner_tl.sum()

    # Top-right corner
    corner_tr = W_full[0:nbc, -nbc:]            # shape = (nbc, nbc)
    g[0, -1] += corner_tr.sum()

    # Bottom-left corner
    corner_bl = W_full[-nbc:, 0:nbc]            # shape = (nbc, nbc)
    g[-1, 0] += corner_bl.sum()

    # Bottom-right corner
    corner_br = W_full[-nbc:, -nbc:]            # shape = (nbc, nbc)
    g[-1, -1] += corner_br.sum()

    return g

def label_thresholded_components(A: np.ndarray,
                                   X: float,
                                   connectivity: int = 4):
    """
    Label all components in A where two pixels are neighbors
    (4- or 8-connectivity) and their absolute difference ≤ X.
    
    Returns
    -------
    labels : np.ndarray of shape A.shape
        Integer labels 0…n_labels-1
    n_labels : int
        Number of connected components found
    """
    H, W = A.shape
    N = H*W

    # 1) build a sparse adjacency matrix of the grid
    adj = lil_matrix((N, N), dtype=bool)
    # choose neighbor offsets
    if connectivity == 4:
        neigh = [(1,0),(-1,0),(0,1),(0,-1)]
    else:  # 8-connectivity
        neigh = [(1,0),(-1,0),(0,1),(0,-1),
                 (1,1),(1,-1),(-1,1),(-1,-1)]

    # 2) for each pixel, connect to any neighbor whose diff ≤ X
    for i in range(H):
        for j in range(W):
            idx = i*W + j
            for di, dj in neigh:
                ni, nj = i+di, j+dj
                if 0 <= ni < H and 0 <= nj < W:
                    if abs(A[ni, nj] - A[i, j]) <= X:
                        adj[idx, ni*W + nj] = True

    # 3) find the connected components in that graph
    adj = adj.tocsr()
    n_labels, flat_labels = connected_components(adj,
                                                directed=False,
                                                return_labels=True)

    # 4) reshape back to H×W
    return flat_labels.reshape(H, W), n_labels

from typing import Optional, Union

import torch
from torch import Tensor

from torch.optim.optimizer import Optimizer, ParamsT

def _strong_wolfe(
    obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9, tolerance_change=1e-9, max_ls=25
):
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev, f_prev, gtd_prev, t, f_new, gtd_new, bounds=(min_step, max_step)
        )

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)  # type: ignore[possibly-undefined]
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:  # type: ignore[possibly-undefined]
            break

        # compute new trial value
        t = _cubic_interpolate(
            bracket[0],
            bracket_f[0],
            bracket_gtd[0],  # type: ignore[possibly-undefined]
            bracket[1],
            bracket_f[1],
            bracket_gtd[1],
        )

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)  # type: ignore[possibly-undefined]
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]  # type: ignore[possibly-undefined]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)  # type: ignore[possibly-undefined]
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]  # type: ignore[possibly-undefined]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]  # type: ignore[possibly-undefined]
    return f_new, g_new, t, ls_func_evals

def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.0


@torch.no_grad()
def bfgs(cost_and_gradient_func, x0, max_iter, tolerance_grad):

    def _directional_evaluate(x, t, d):
        xx = x+t*d
        return cost_and_gradient_func(xx)

    def _add_grad(x,t,d):
        x+=t*d

    lr = 1
    max_eval = max_iter * 5 // 4

    # evaluate initial f(x) and df/dx
    orig_loss, flat_grad = cost_and_gradient_func(x0)
    x=x0
    loss = float(orig_loss)
    current_evals = 1
    opt_cond = flat_grad.abs().max() <= tolerance_grad

    # optimal condition
    if opt_cond:
        return x


    n_iter = 0
    al_made = False
    prev_flat_grad = None
    cur_index = 0
    ro = torch.zeros(max_iter+1, dtype = torch.float64, device='cuda')
    al = torch.zeros(max_iter+1, dtype = torch.float64, device='cuda')
    #be = torch.zeros(max_iter+1, dtype = torch.float64, device='cuda')
    old_dirs = torch.zeros(max_iter+1,x.shape[0], dtype =  torch.float64, device='cuda')
    old_stps = torch.zeros(max_iter+1,x.shape[0], dtype =  torch.float64, device='cuda')
    # optimize for a max of max_iter iterations
    while n_iter < max_iter:
        # keep track of nb of iterations
        n_iter += 1

        ############################################################
        # compute gradient descent direction
        ############################################################
        if n_iter == 1:
            d = flat_grad.neg()
            #old_dirs = []
            #old_stps = []
            #ro = []
            H_diag = 1
        else:
            # do lbfgs update (update memory)
            y = flat_grad.sub(prev_flat_grad)
            s = d.mul(t)
            ys = y.dot(s)  # y*s
            if ys > 1e-10:
                # updating memory
                # if len(old_dirs) == history_size:
                #     # shift history by one (limited-memory)
                #     old_dirs.pop(0)
                #     old_stps.pop(0)
                #     ro.pop(0)

                # store new direction/step
                old_dirs[cur_index,:] = y
                old_stps[cur_index,:] = s
                ro[cur_index] = 1.0/ys;
                cur_index+=1
                #old_dirs.append(y)
                #old_stps.append(s)
                #ro.append(1.0 / ys)

                # update scale of initial Hessian approximation
                H_diag = ys / y.dot(y)  # (y*y)

            # compute the approximate (L-BFGS) inverse Hessian
            # multiplied by the gradient
            num_old = cur_index            
            q = flat_grad.neg()
            # for i in range(num_old - 1, -1, -1):
            #     al[i] = old_stps[i,:].dot(q) * ro[i]
            #     q.add_(old_dirs[i,:], alpha=-al[i])
            lbfgs_backward_torch(old_stps,
                         old_dirs,
                         ro,                         
                         q, al, num_old)

            # multiply by initial Hessian
            # r/d is the final direction
            d = r = torch.mul(q, H_diag)
            #for i in range(num_old):
            #    be_i = old_dirs[i,:].dot(r) * ro[i]
            #    r.add_(old_stps[i,:], alpha=al[i] - be_i)            
            lbfgs_forward_torch(old_stps,
                         old_dirs,
                         ro,                         
                         r, al, num_old)

        if prev_flat_grad is None:
            prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
        else:
            prev_flat_grad.copy_(flat_grad)
        prev_loss = loss

        ############################################################
        # compute step length
        ############################################################
        # reset initial guess for step size
        if n_iter == 1:
            t = min(1.0, 1.0 / flat_grad.abs().sum()) * lr
        else:
            t = lr

        # directional derivative
        gtd = flat_grad.dot(d)  # g * d

        # optional line search: user function
        ls_func_evals = 0
        x_init = copy.deepcopy(x)

        def obj_func(x, t, d):
            return _directional_evaluate(x, t, d)

        loss, flat_grad, t, ls_func_evals = _strong_wolfe(
            obj_func, x_init, t, d, loss, flat_grad, gtd
        )
        _add_grad(x, t, d)
        opt_cond = flat_grad.abs().max() <= tolerance_grad
        #if n_iter%500==0:
        #    print(n_iter, flat_grad.abs().max().detach().cpu().numpy())
       
        # update func eval
        current_evals += ls_func_evals
        
        ############################################################
        # check conditions
        ############################################################
        if n_iter == max_iter:
            break

        if current_evals >= max_eval:
            break

        # optimal condition
        if opt_cond:
            break

    return x

kernel_code = r'''
extern "C" __global__
void lbfgs_backward(const double* __restrict__ old_stps,
                    const double* __restrict__ old_dirs,
                    const double* __restrict__ ro,
                    double*       __restrict__ q,
                    double*       __restrict__ al,
                    int num_old,
                    int n) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int chunk = (n + block_size - 1) / block_size;
    

    for (int i = num_old - 1; i >= 0; --i) {
        double local_sum = 0.0f;
        int base = tid;
        for (int c = 0; c < chunk; ++c) {
            int idx = base + c * block_size;
            if (idx < n) {
                local_sum += old_stps[i * n + idx] * q[idx];
            }
        }
        sdata[tid] = local_sum;
        __syncthreads();

        // tree-reduction in shared memory
        for (int stride = block_size >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sdata[tid] += sdata[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            
            sdata[0] = ro[i] * sdata[0];
            al[i] = sdata[0];
        }
        __syncthreads();

        double ai = sdata[0];
        for (int c = 0; c < chunk; ++c) {
            int idx = base + c * block_size;
            if (idx < n) {
                q[idx] -= ai * old_dirs[i * n + idx];
            }
        }
        __syncthreads();
    }
}
'''

module = cp.RawModule(code=kernel_code,
                      options=('--std=c++11',),
                      name_expressions=('lbfgs_backward',))
lbfgs_backward = module.get_function('lbfgs_backward')


kernel_code = r'''
extern "C" __global__
void lbfgs_forward(const double* __restrict__ old_stps,
                    const double* __restrict__ old_dirs,
                    const double* __restrict__ ro,
                    double*       __restrict__ r,
                    double*       __restrict__ al,
                    int num_old,
                    int n) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int chunk = (n + block_size - 1) / block_size;
    

    for (int i = 0; i <= num_old; i++) {
        double local_sum = 0.0f;
        int base = tid;
        for (int c = 0; c < chunk; ++c) {
            int idx = base + c * block_size;
            if (idx < n) {
                local_sum += old_dirs[i * n + idx] * r[idx];
            }
        }
        sdata[tid] = local_sum;
        __syncthreads();

        // tree-reduction in shared memory
        for (int stride = block_size >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sdata[tid] += sdata[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {            
            sdata[0] = ro[i] * sdata[0];
        }
        __syncthreads();

        double be_i = sdata[0];
        for (int c = 0; c < chunk; ++c) {
            int idx = base + c * block_size;
            if (idx < n) {
                r[idx] += (al[i]-be_i) * old_stps[i * n + idx];
            }
        }
        __syncthreads();
    }
}
'''

module = cp.RawModule(code=kernel_code,
                      options=('--std=c++11',),
                      name_expressions=('lbfgs_forward',))
lbfgs_forward = module.get_function('lbfgs_forward')


def lbfgs_backward_torch(old_stps: torch.Tensor,
                         old_dirs: torch.Tensor,
                         ro: torch.Tensor,
                         q: torch.Tensor,
                         al: torch.Tensor, m):
    """
    In-place backward L-BFGS loop on CUDA tensors:
        for i in range(m-1, -1, -1):
            al[i] = dot(old_stps[i], q) * ro[i]
            q     -= al[i] * old_dirs[i]
    All inputs must be float64 CUDA tensors.
    old_stps, old_dirs: (m, n)
    ro, al:             (m,)
    q:                  (n,)
    """
    # sanity checks
    assert old_stps.is_cuda and old_dirs.is_cuda and ro.is_cuda and q.is_cuda
    assert old_stps.dtype == torch.float64 and q.dtype == torch.float64
    _, n = old_stps.shape


    # zero-copy conversion to CuPy via DLPack
    old_stps_c = cp.from_dlpack(to_dlpack(old_stps))
    old_dirs_c = cp.from_dlpack(to_dlpack(old_dirs))
    ro_c       = cp.from_dlpack(to_dlpack(ro))
    q_c        = cp.from_dlpack(to_dlpack(q))
    al_c       = cp.from_dlpack(to_dlpack(al))

    # launch parameters
    block_size = 1024
    grid_size  = 1
    shared_mem_bytes = block_size * np.dtype('float64').itemsize

    # invoke the kernel
    lbfgs_backward(
        (grid_size,), (block_size,),
        (old_stps_c, old_dirs_c, ro_c, q_c, al_c,
         np.int32(m), np.int32(n)),
        shared_mem=shared_mem_bytes
    )

def lbfgs_forward_torch(old_stps: torch.Tensor,
                         old_dirs: torch.Tensor,
                         ro: torch.Tensor,
                         r: torch.Tensor,
                         al: torch.Tensor, m):

    # sanity checks
    assert old_stps.is_cuda and old_dirs.is_cuda and ro.is_cuda and r.is_cuda
    assert old_stps.dtype == torch.float64 and r.dtype == torch.float64
    _, n = old_stps.shape


    # zero-copy conversion to CuPy via DLPack
    old_stps_c = cp.from_dlpack(to_dlpack(old_stps))
    old_dirs_c = cp.from_dlpack(to_dlpack(old_dirs))
    ro_c       = cp.from_dlpack(to_dlpack(ro))
    r_c        = cp.from_dlpack(to_dlpack(r))
    al_c       = cp.from_dlpack(to_dlpack(al))

    # launch parameters
    block_size = 1024
    grid_size  = 1
    shared_mem_bytes = block_size * np.dtype('float64').itemsize

    # invoke the kernel
    lbfgs_forward(
        (grid_size,), (block_size,),
        (old_stps_c, old_dirs_c, ro_c, r_c, al_c,
         np.int32(m), np.int32(n)),
        shared_mem=shared_mem_bytes
    )

