'''
Implements the vel_to_seis forward model, including forward and backward gradient propagation.
Based on the original MATLAB implementation here: https://csim.kaust.edu.sa/files/SeismicInversion/Chapter.FD/lab.FD2.8/lab.html
'''

import numpy as np
import cupy as cp
import kaggle_support as kgs
import copy
import seis_numerics
import time

# Set up stuff for CUDA graphing
stream = cp.cuda.Stream(non_blocking=True)
graph = 0
graph_diff = 0
graph_adjoint = 0

# Precalculations, including preallocating matrices. The biggest matrices will only be allocated on demand in vel_to_seis.
def ricker(f, dt, nt=None):
    nw = int(2.2 / f / dt)
    nw = 2 * (nw // 2) + 1
    nc = nw // 2 + 1 
    k = np.arange(1, nw + 1)  
    alpha = (nc - k) * f * dt * np.pi
    beta = alpha ** 2
    w0 = (1.0 - 2.0 * beta) * np.exp(-beta)    
    if nt is not None:
        if nt < len(w0):
            raise ValueError("nt is smaller than condition!")
        w = np.zeros(nt)  
        w[0:len(w0)] = w0
    else:
        w = np.zeros(len(w0))
        w[0:] = w0
    if nt is not None:
        tw = np.arange(1, len(w)) * dt
    else:
        tw = np.arange(1, len(w)) * dt
    return w, tw
def expand_source(s0, nt):
    s0 = np.asarray(s0).flatten()
    s = np.zeros(nt)
    s[0:len(s0)] = s0
    return s
def AbcCoef2D(nzbc, nxbc, nbc, dx):
    nz = nzbc - 2 * nbc
    nx = nxbc - 2 * nbc
    a = (nbc - 1) * dx
    kappa = 3.0 * np.log(1e7) / (2.0 * a)
    damp1d = kappa * (((np.arange(1, nbc + 1) - 1) * dx / a) ** 2)
    damp = np.zeros((nzbc, nxbc))
    for iz in range(nzbc):
        damp[iz, :nbc] = damp1d[::-1]
        damp[iz, nx + nbc : nx + 2 * nbc] = damp1d
    for ix in range(nbc, nbc + nx):
        damp[:nbc, ix] = damp1d[::-1]
        damp[nz + nbc: nz + 2 * nbc, ix] = damp1d
    return cp.array(damp, dtype = kgs.base_type_gpu)       	
def adjust_sr(coord, dx, nbc):
    isx = int(round(coord['sx'] / dx)) + nbc
    isz = int(round(coord['sz'] / dx)) + nbc
    igx = (np.round(np.array(coord['gx']) / dx) + nbc).astype(int)
    igz = (np.round(np.array(coord['gz']) / dx) + nbc).astype(int)
    if abs(coord['sz']) < 0.5:
        isz += 1
    igz = igz + (np.abs(np.array(coord['gz'])) < 0.5).astype(int)
    return isx, isz, igx, igz
nz = 70
nx = 70
dx = 10
nbc = 120
nt = 999
dt = (1e-3)
freq = 15
s, _ = (ricker(freq, dt))
s = expand_source(s, nt)
s = cp.array(s, dtype=kgs.base_type_gpu)
c1 = (-2.5)
c2 = (4.0 / 3.0)
c3 = (-1.0 / 12.0)
c2,c3 = np.array(c2,dtype=kgs.base_type), np.array(c3,dtype=kgs.base_type)
src_idx_list = []
isx_list = []
isz_list = []
for i_source in range(5):
    coord = {}
    source_x = [0, 17, 34, 52, 69][i_source]
    coord['sx'] = source_x * dx        
    coord['sz'] = 1 * dx
    coord['gx'] = np.arange(0, nx) * dx
    coord['gz'] = np.ones_like(coord['gx']) * dx
    isx, isz, igx, igz = adjust_sr(coord, dx, nbc)
    src_idx = np.int32(isz*310 + isx)
    src_idx_list.append(src_idx)
    isx_list.append(isx)
    isz_list.append(isz)
ng = len(coord['gx'])
damp = AbcCoef2D(310,310, nbc, dx)
nx,nz = 310,310
rcv_idx = nx*igz+igx
# Prepare base matrices - first few are done on demand later
seis_combined = None#cp.zeros((5,999,70),dtype=kgs.base_type_gpu)
p_complete = None#cp.zeros((nt+2,nx,nz), dtype=kgs.base_type_gpu)
lapg_store = None#cp.zeros((nt,nx,nz), dtype=kgs.base_type_gpu)
temp1 = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
temp2 = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
alpha = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
v = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
temp1_flat = temp1.ravel();temp2_flat = temp2.ravel();alpha_flat = alpha.ravel()        
s_mod = cp.zeros_like(s)
# Prepare forward propagation matrices - first few are done on demand later
seis_combined_diff = None#cp.zeros((5,999,70),dtype=kgs.base_type_gpu)
p_complete_diff = None#cp.zeros((nt+2,nx,nz), dtype=kgs.base_type_gpu)
lapg_store_diff = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
temp1_diff = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
temp2_diff = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
alpha_diff = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
v_diff = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
s_mod_diff = cp.zeros_like(s)
lapg_store_diff_flat = lapg_store_diff.ravel()
temp1_diff_flat = temp1_diff.ravel()
temp2_diff_flat = temp2_diff.ravel()
alpha_diff_flat = alpha_diff.ravel()
v_diff_flat = v_diff.ravel()
# Prepare backward propagation matrices - first few are done on demand later
seis_combined_adjoint = None#cp.zeros((5,999,70),dtype=kgs.base_type_gpu)
p_complete_adjoint = None#cp.zeros((nt+2,nx,nz), dtype=kgs.base_type_gpu)
lapg_store_adjoint = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
temp1_adjoint = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
temp2_adjoint = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
alpha_adjoint = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
v_adjoint = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
s_mod_adjoint = cp.zeros_like(s)
lapg_store_adjoint_flat = lapg_store_adjoint.ravel()
temp1_adjoint_flat = temp1_adjoint.ravel()
temp2_adjoint_flat = temp2_adjoint.ravel()
alpha_adjoint_flat = alpha_adjoint.ravel()
v_adjoint_flat = v_adjoint.ravel()

src_idx_dev = cp.zeros((1,), dtype=cp.int32)


igz_dev,igx_dev = cp.array(igz), cp.array(igx)

def vel_to_seis(vec, vec_diff=None, vec_adjoint=None, adjoint_on_residual=False):
    # Outputs:
    # result: the seismogram associated with velocity field vec
    # result_diff: J@vec_diff, where J is the Jacobian of the operation above
    # result_adjoint: J^T@vec_adjoint, or J^T@(result-vec_adjoint) if adjoint_on_residual=True

    # Input checks
    assert vec.shape == (4901,1)
    assert vec_adjoint is None or vec_adjoint.shape == (5*999*70,1)
    assert vec_diff is None or vec_diff.shape == (4901,1)
    do_diff = not (vec_diff is None)
    do_adjoint = not (vec_adjoint is None)
    assert vec.dtype == kgs.base_type_gpu
    assert vec_diff is None or vec_diff.dtype == kgs.base_type_gpu
    assert vec_adjoint is None or vec_adjoint.dtype == kgs.base_type_gpu

    # Preallocate matrices not done above
    global seis_combined, p_complete, lapg_store, seis_combined_diff, p_complete_diff, seis_combined_adjoint, p_complete_adjoint
    if seis_combined is None:
        seis_combined = cp.zeros((5,999,70),dtype=kgs.base_type_gpu)
        p_complete = cp.zeros((nt+2,nx,nz), dtype=kgs.base_type_gpu)
        lapg_store = cp.zeros((nt,nx,nz), dtype=kgs.base_type_gpu)
        seis_combined_diff = cp.zeros((5,999,70),dtype=kgs.base_type_gpu)
        p_complete_diff = cp.zeros((nt+2,nx,nz), dtype=kgs.base_type_gpu)
        seis_combined_adjoint = cp.zeros((5,999,70),dtype=kgs.base_type_gpu)
        p_complete_adjoint = cp.zeros((nt+2,nx,nz), dtype=kgs.base_type_gpu)
        
        p_complete_flat = p_complete.ravel();lapg_store_flat=lapg_store.ravel()
        p_complete_diff_flat = p_complete_diff.ravel()
        p_complete_adjoint_flat = p_complete_adjoint.ravel()
    
    # PREPARATION
    prep_run(vec)
     
    if do_diff:
        prep_run_diff(vec_diff)        

    if do_adjoint:        
        alpha_adjoint[...] = 0
        temp1_adjoint[...] = 0
        temp2_adjoint[...] = 0
        v_adjoint[...] = 0
        seis_combined_adjoint[...] = cp.reshape(vec_adjoint, (5,999,70))

    tx, ty = 32, 32
    bx = (nx + tx - 1) // tx
    by = (nz + ty - 1) // ty  

    # LOOP

    global stream, graph, graph_adjoint, graph_diff

    cp.cuda.Stream.null.synchronize()

    with stream:        
        for i_source in range(5):     
            ## Calculate seismogram
            # Prepare source injection
            src_idx = src_idx_list[i_source]
            bdt = (v[isz_list[i_source], isx_list[i_source]]*dt)**2
            s_mod[...] = bdt*s            
            src_idx_dev[...] = cp.array(src_idx, dtype=cp.int32)

            # We capture the full time loop as a CUDA graph.
            if graph==0:
                stream.begin_capture()
                for it in range(0, nt):
                    # Code below is equivalent to:
                    # p1 = p_complete[it+1,...]
                    # p0 = p_complete[it,...]
                    # lapg_store[it+2,...] = (cp.array(c2) * (cp.roll(p1, 1, axis=1) + cp.roll(p1, -1, axis=1) +
                    #                cp.roll(p1, 1, axis=0) + cp.roll(p1, -1, axis=0)) +
                    #          cp.array(c3) * (cp.roll(p1, 2, axis=1) + cp.roll(p1, -2, axis=1) +
                    #                cp.roll(p1, 2, axis=0) + cp.roll(p1, -2, axis=0)))
                    # p_complete[it+2,...] = (temp1 * p1 - temp2 * p0 +
                    #      alpha * lapg_store[it+2,...])
                    # p_complete[it+2,...].ravel()[src_idx] += s_mod[it]   
                    lapg(
                        (bx, by), (tx, ty),
                        (p_complete_flat[(it+1)*(nx*nz):],
                        lapg_store_flat[nx*nz*it:],
                        nx, nz,
                        c2, c3,
                        ))     
                    update_p(
                                (bx, by), (tx, ty),
                                (
                                    temp1_flat, temp2_flat, alpha_flat,
                                    p_complete_flat[it*(nx*nz):],
                                    p_complete_flat[(it+1)*(nx*nz):],
                                    p_complete_flat[(it+2)*(nx*nz):],
                                    lapg_store_flat[nx*nz*it:],
                                    nx, nz, it,
                                    src_idx_dev, s_mod
                                )
                            )
    
                graph = stream.end_capture()
                graph.upload(stream)
            graph.launch(stream)
            stream.synchronize()
            # Store results
            seis_combined[i_source,...] = p_complete[2:,igz_dev,igx_dev]

            ## Forward propagation
            if do_diff:                
                bdt_diff = 2*((v[isz_list[i_source], isx_list[i_source]]*v_diff[isz_list[i_source], isx_list[i_source]]))* dt**2
                s_mod_diff[...] = bdt_diff*s
                if graph_diff==0:
                    stream.begin_capture()                    
                    for it in range(0, nt):
                        lapg(
                            (bx, by), (tx, ty),
                            (p_complete_diff_flat[(it+1)*(nx*nz):],
                            lapg_store_diff,
                            nx, nz,
                            c2, c3,
                            ))
                        update_p_diff(
                                    (bx, by), (tx, ty),
                                    (
                                        temp1_flat, temp1_diff_flat, temp2_flat, temp2_diff_flat, alpha_flat, alpha_diff_flat, 
                                        p_complete_flat[it*(nx*nz):],
                                        p_complete_flat[(it+1)*(nx*nz):],
                                        p_complete_flat[(it+2)*(nx*nz):],
                                        lapg_store_flat[nx*nz*it:], lapg_store_diff_flat,
                                        p_complete_diff_flat[it*(nx*nz):],
                                        p_complete_diff_flat[(it+1)*(nx*nz):],
                                        p_complete_diff_flat[(it+2)*(nx*nz):],
                                        nx, nz, it,
                                        src_idx_dev, s_mod_diff
                                    )
                                )
                    graph_diff = stream.end_capture()
                    graph_diff.upload(stream)       
                graph_diff.launch(stream)
                seis_combined_diff[i_source,...] = p_complete_diff[2:,igz_dev,igx_dev]
            # Backward propagation
            if do_adjoint:     
                p_complete_adjoint[...] = 0
                if adjoint_on_residual:
                    p_complete_adjoint[2:,igz_dev,igx_dev] = seis_combined[i_source,...]-seis_combined_adjoint[i_source,...]
                else:
                    p_complete_adjoint[2:,igz_dev,igx_dev] = seis_combined_adjoint[i_source,...]
                s_mod_adjoint[...] = 0
                if graph_adjoint==0:
                    stream.begin_capture()
                    print('capturing graph')
                    for it in np.arange(nt-1,-1,-1):                               
                        update_p_adjoint(
                                (bx, by), (tx, ty),
                                (
                                    temp1_flat, temp2_flat, alpha_flat,
                                    p_complete_flat[(it)*(nx*nz):],p_complete_flat[(it+1)*(nx*nz):],
                                    lapg_store_flat[(it)*(nx*nz):],
                                    s_mod_adjoint,  
                                    p_complete_adjoint_flat[(it)*(nx*nz):],
                                    p_complete_adjoint_flat[(it+1)*(nx*nz):],
                                    p_complete_adjoint_flat[(it+2)*(nx*nz):],
                                    temp1_adjoint_flat, temp2_adjoint_flat, alpha_adjoint_flat, lapg_store_adjoint_flat,                                    
                                    nx, nz, it,
                                    c2, c3,
                                    src_idx_dev
                                )
                            )
                    graph_adjoint = stream.end_capture()
                    graph_adjoint.upload(stream)  
                graph_adjoint.launch(stream)
                v_adjoint[isz_list[i_source], isx_list[i_source]] += 2*dt**2 * v[isz_list[i_source], isx_list[i_source]] * cp.sum(s_mod_adjoint*s)
    
    stream.synchronize()
            
    # FINALIZE
    assert seis_combined.shape == (5,999,70)
    result =  cp.copy(seis_combined.flatten()[:,None])
    if do_diff:
        assert seis_combined_diff.shape == (5,999,70)
        result_diff =  cp.copy(seis_combined_diff.flatten()[:,None])
    else:
        result_diff = None
    if do_adjoint:
        result_adjoint = cp.copy(prep_run_adjoint())
    else:
        result_adjoint = None

    return result, result_diff, result_adjoint

def prep_run(vec):

    vv=cp.reshape(vec[:-1,0], (70,70))
    min_vel = vec[-1,0]
    
    v[...] = cp.pad(vv, ((nbc, nbc), (nbc, nbc)), mode='edge')
    abc = min_vel*damp

    alpha[...] = (v * (dt / dx)) ** 2    
    kappa = abc * dt
    temp1[...] = 2 + 2 * c1 * alpha - kappa
    temp2[...] = 1 - kappa

def prep_run_diff(vec_diff):

    vv=cp.reshape(vec_diff[:-1,0], (70,70))
    min_vel_diff = vec_diff[-1,0]
    
    v_diff[...] = cp.pad(vv, ((nbc, nbc), (nbc, nbc)), mode='edge')
    abc_diff = min_vel_diff*damp

    alpha_diff[...] = v_diff * v * (2*(dt / dx) **2)
    kappa_diff = abc_diff * dt
    temp1_diff[...] = 2 * c1 * alpha_diff - kappa_diff
    temp2_diff[...] = - kappa_diff

    return v_diff,temp1_diff,temp2_diff,alpha_diff

def prep_run_adjoint():

    kappa_adjoint = -temp2_adjoint
    alpha_adjoint[...] += 2 * c1 * temp1_adjoint
    kappa_adjoint += -temp1_adjoint
    abc_adjoint = kappa_adjoint * dt
    v2_adjoint = alpha_adjoint * (dt/dx)**2

    v_adjoint[...] += 2*v*v2_adjoint
    min_vel_adjoint = cp.sum(abc_adjoint*damp)
    vv_adjoint = seis_numerics.unpad_edge_padded_gradient(v_adjoint,nbc)#v_adjoint[nbc:-nbc,nbc:-nbc]

    result_adjoint = cp.zeros((4901,1),dtype=kgs.base_type_gpu)
    result_adjoint[-1,0] = min_vel_adjoint
    result_adjoint[:-1,0] = vv_adjoint.flatten()

    return result_adjoint


# CUDA kernel to compute Laplacian
kernel_code = r'''
extern "C" __global__
void lapg(
              const floattype* __restrict__ input,
              floattype* __restrict__ output,
              const int    nx,
              const int    ny,
              const floattype  c2,
              const floattype  c3) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;
    int idx = iy * nx + ix;

    // Manual wrap at ±1, ±2
    int ix_p1 = ix+1; if (ix_p1==nx)  ix_p1=0;
    int ix_m1 = ix-1; if (ix_m1<0)    ix_m1=nx-1;
    int ix_p2 = ix+2; if (ix_p2>=nx)  ix_p2-=nx;
    int ix_m2 = ix-2; if (ix_m2<0)     ix_m2+=nx;
    int iy_p1 = iy+1; if (iy_p1==ny)  iy_p1=0;
    int iy_m1 = iy-1; if (iy_m1<0)    iy_m1=ny-1;
    int iy_p2 = iy+2; if (iy_p2>=ny)  iy_p2-=ny;
    int iy_m2 = iy-2; if (iy_m2<0)     iy_m2+=ny;

    floattype t1;
    floattype t2;
    
    // Collect neighbors (±1)
    t1 = input[iy  * nx + ix_p1]
             + input[iy  * nx + ix_m1]
             + input[iy_p1 * nx + ix  ]
             + input[iy_m1 * nx + ix  ];
    // Collect neighbors (±2)
    t2 = input[iy  * nx + ix_p2]
             + input[iy  * nx + ix_m2]
             + input[iy_p2 * nx + ix  ]
             + input[iy_m2 * nx + ix  ];

    
    output[idx] = c2*t1+c3*t2;
}
'''

module = cp.RawModule(code=kernel_code.replace('floattype', kgs.base_type_str))
lapg = module.get_function('lapg')

# CUDA kernel to update p and add source
kernel_code = r'''
extern "C" __global__
void update_p(
              const floattype* __restrict__ temp1,
              const floattype* __restrict__ temp2,
              const floattype* __restrict__ alpha,
              const floattype*             __restrict__ pout,
              const floattype*             __restrict__ pout1,
              floattype*             __restrict__ pout2,
              const floattype*             __restrict__ lapg_store,
              const int    nx,
              const int    ny,
              const int    it,
              const int*   __restrict__ src_idx,
              const floattype* __restrict__ s_mod) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;
    int idx = iy * nx + ix;

    floattype temp1_local = __ldg(&temp1[idx]);
    floattype temp2_local = __ldg(&temp2[idx]);
    floattype pout1_local = __ldg(&pout1[idx]);
    floattype pout_local = __ldg(&pout[idx]);
    floattype alpha_local = __ldg(&alpha[idx]);
    floattype out = temp1_local*pout1_local
              - temp2_local*pout_local
              + alpha_local*lapg_store[idx];

    // fused source injection:
    if (idx == src_idx[0]) {
        out += s_mod[it];
    }
    pout2[idx] = out;
}
'''

module = cp.RawModule(code=kernel_code.replace('floattype', kgs.base_type_str))
update_p = module.get_function('update_p')

# Forward propagation of update_p
kernel_code = r'''
extern "C" __global__
void update_p_diff(
              const floattype* __restrict__ temp1,
              const floattype* __restrict__ temp1_diff,
              const floattype* __restrict__ temp2,
              const floattype* __restrict__ temp2_diff,
              const floattype* __restrict__ alpha,
              const floattype* __restrict__ alpha_diff,
              const floattype*             __restrict__ pout,
              const floattype*             __restrict__ pout1,
              const floattype*             __restrict__ pout2,
              const floattype*             __restrict__ lapg_store,
              const floattype*             __restrict__ lapg_store_diff,
              floattype*  __restrict__ pout_diff,
              floattype*  __restrict__ pout1_diff,
              floattype*  __restrict__ pout2_diff,
              const int    nx,
              const int    ny,
              const int it,
              const int*   __restrict__ src_idx,
              const floattype* __restrict__ s_mod_diff) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;
    int idx = iy * nx + ix;

    floattype out = temp1_diff[idx] * pout1[idx] + temp1[idx]*pout1_diff[idx]-temp2_diff[idx] * pout[idx] - temp2[idx]*pout_diff[idx] +
        alpha_diff[idx]*lapg_store[idx] + alpha[idx]*lapg_store_diff[idx];

    //floattype out = temp1_local*pout1_local
    //          - temp2_local*pout_local
    //          + alpha_local*lapg_store_local;

    //floattype out = 0
    // fused source injection:
    if (idx == src_idx[0]) {
        out += s_mod_diff[it];
    }
    pout2_diff[idx] = out;
}
'''

module = cp.RawModule(code=kernel_code.replace('floattype', kgs.base_type_str))
update_p_diff = module.get_function('update_p_diff')

# Backward propagation of update_p (with lapg also folded in)
kernel_code = r'''
extern "C" __global__
void update_p_adjoint(
              const floattype* __restrict__ temp1,
              const floattype* __restrict__ temp2,
              const floattype* __restrict__ alpha,
              const floattype* __restrict__ p_complete1,
              const floattype* __restrict__ p_complete2,
              const floattype* __restrict__ lapg_store,
              floattype*  __restrict__ s_mod_adjoint,
              floattype*  __restrict__ p_complete_adjoint1,
              floattype*  __restrict__ p_complete_adjoint2,
              floattype*  __restrict__ p_complete_adjoint3,
              floattype*  __restrict__ temp1_adjoint,
              floattype*  __restrict__ temp2_adjoint,
              floattype*  __restrict__ alpha_adjoint,
              floattype*  __restrict__ lapg_store_adjoint,
              const int    nx,
              const int    ny,
              const int    it,
              const floattype  c2,
              const floattype  c3,
              const int*   __restrict__ src_idx) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;
    int idx = iy * nx + ix;


    if (idx == src_idx[0]) {
        s_mod_adjoint[it] = p_complete_adjoint3[idx];
    }

    p_complete_adjoint2[idx] += temp1[idx]*p_complete_adjoint3[idx];
    temp1_adjoint[idx] += p_complete2[idx]*p_complete_adjoint3[idx];
    p_complete_adjoint1[idx] -= temp2[idx]*p_complete_adjoint3[idx];
    temp2_adjoint[idx] -= p_complete1[idx]*p_complete_adjoint3[idx];
    alpha_adjoint[idx] += lapg_store[idx] * p_complete_adjoint3[idx];
    //lapg_store_adjoint[idx] = alpha[idx] * p_complete_adjoint3[idx];

    // Manual wrap at ±1, ±2
    int ix_p1 = ix+1; if (ix_p1==nx)  ix_p1=0;
    int ix_m1 = ix-1; if (ix_m1<0)    ix_m1=nx-1;
    int ix_p2 = ix+2; if (ix_p2>=nx)  ix_p2-=nx;
    int ix_m2 = ix-2; if (ix_m2<0)     ix_m2+=nx;
    int iy_p1 = iy+1; if (iy_p1==ny)  iy_p1=0;
    int iy_m1 = iy-1; if (iy_m1<0)    iy_m1=ny-1;
    int iy_p2 = iy+2; if (iy_p2>=ny)  iy_p2-=ny;
    int iy_m2 = iy-2; if (iy_m2<0)     iy_m2+=ny;

    // Collect neighbors (±1)
    floattype t1 = alpha[iy  * nx + ix_p1] * p_complete_adjoint3[0 + iy  * nx + ix_p1] +
                alpha[iy  * nx + ix_m1] * p_complete_adjoint3[0 + iy  * nx + ix_m1] +
                alpha[iy_p1  * nx + ix] * p_complete_adjoint3[0 + iy_p1  * nx + ix] +
                alpha[iy_m1  * nx + ix] * p_complete_adjoint3[0 + iy_m1  * nx + ix];
    // Collect neighbors (±2)
    floattype t2 = alpha[iy  * nx + ix_p2] * p_complete_adjoint3[0 + iy  * nx + ix_p2] +
                alpha[iy  * nx + ix_m2] * p_complete_adjoint3[0 + iy  * nx + ix_m2] +
                alpha[iy_p2  * nx + ix] * p_complete_adjoint3[0 + iy_p2  * nx + ix] +
                alpha[iy_m2  * nx + ix] * p_complete_adjoint3[0 + iy_m2  * nx + ix];

    p_complete_adjoint2[idx]+=c2*t1+c3*t2;
      
}
'''
module = cp.RawModule(code=kernel_code.replace('floattype', kgs.base_type_str))
update_p_adjoint = module.get_function('update_p_adjoint')









# Precompute stuff
