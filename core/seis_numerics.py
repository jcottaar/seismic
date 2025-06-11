import cupy as cp
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import lil_matrix

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