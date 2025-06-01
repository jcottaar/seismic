# Original source: XXX
# Python translation adapted from: https://www.kaggle.com/code/jaewook704/waveform-inversion-vel-to-seis by Jae-Wook Kim

import numpy as np
import cupy as cp
import kaggle_support as kgs
import copy
import time
import glob

reference_mode = False # if true will use simpler expressions

# CUDA kernel to update p and add source
kernel_code = r'''
extern "C" __global__
void update_p(
              const double* __restrict__ temp1,
              const double* __restrict__ temp2,
              const double* __restrict__ alpha,
              double*             __restrict__ pout,
              double*             __restrict__ lapg_store,
              const int    ind_offset1,
              const int    ind_offset2,
              const int    ind_offset3,
              const int    nx,
              const int    ny,
              const double  c2,
              const double  c3,
              const int   src_idx,
              const double s_val) {
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

    // Collect neighbors (±1)
    double t1 = pout[ind_offset2 + iy  * nx + ix_p1]
             + pout[ind_offset2 + iy  * nx + ix_m1]
             + pout[ind_offset2 + iy_p1 * nx + ix  ]
             + pout[ind_offset2 + iy_m1 * nx + ix  ];
    // Collect neighbors (±2)
    double t2 = pout[ind_offset2 + iy  * nx + ix_p2]
             + pout[ind_offset2 + iy  * nx + ix_m2]
             + pout[ind_offset2 + iy_p2 * nx + ix  ]
             + pout[ind_offset2 + iy_m2 * nx + ix  ];

    double lapg_store_local = (c2*t1+c3*t2);
    lapg_store[idx] = lapg_store_local;

    double out = temp1[idx]*pout[ind_offset2 + idx] 
              - temp2[idx]*pout[ind_offset1+idx]
              + alpha[idx]*lapg_store_local;

    // fused source injection:
    if (idx == src_idx) {
        out += s_val;
    }
    pout[ind_offset3+idx] = out;
}
'''
module = cp.RawModule(code=kernel_code)
update_p = module.get_function('update_p')

# CUDA kernel to compute Laplacian over all slices of a 3D matrix
kernel_code = r'''
extern "C" __global__
void compute_lapg_per_slice(
              const double* __restrict__ mat,
              double* __restrict__ lapg_store,
              const int    nx,
              const int    ny,
              const int    N,
              const double  c2,
              const double  c3) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;
    int idx = iy * nx + ix;

    int ix_p1 = ix+1; if (ix_p1==nx)  ix_p1=0;
    int ix_m1 = ix-1; if (ix_m1<0)    ix_m1=nx-1;
    int ix_p2 = ix+2; if (ix_p2>=nx)  ix_p2-=nx;
    int ix_m2 = ix-2; if (ix_m2<0)     ix_m2+=nx;
    int iy_p1 = iy+1; if (iy_p1==ny)  iy_p1=0;
    int iy_m1 = iy-1; if (iy_m1<0)    iy_m1=ny-1;
    int iy_p2 = iy+2; if (iy_p2>=ny)  iy_p2-=ny;
    int iy_m2 = iy-2; if (iy_m2<0)     iy_m2+=ny;

    int i_slice, offset;
    double t1,t2;
    for (i_slice=0 ; i_slice<N ; i_slice++) {

        offset = i_slice*nx*ny;
    
        // Collect neighbors (±1)
        t1 = mat[offset+iy  * nx + ix_p1]
                 + mat[offset+iy  * nx + ix_m1]
                 + mat[offset+iy_p1 * nx + ix  ]
                 + mat[offset+iy_m1 * nx + ix  ];
        // Collect neighbors (±2)
        t2 = mat[offset+iy  * nx + ix_p2]
                 + mat[offset+iy  * nx + ix_m2]
                 + mat[offset+iy_p2 * nx + ix  ]
                 + mat[offset+iy_m2 * nx + ix  ];
    
        lapg_store[offset+idx] = (c2*t1+c3*t2);
    }
    
}
'''
module = cp.RawModule(code=kernel_code)
compute_lapg_per_slice = module.get_function('compute_lapg_per_slice')

# Precompute stuff
def ricker(f, dt, nt=None):
    nw = int(2.2 / f / dt)
    nw = 2 * (nw // 2) + 1
    nc = nw // 2 + 1  # 중심 인덱스를 1-based 기준으로 설정

    k = np.arange(1, nw + 1)  # 1-based index
    alpha = (nc - k) * f * dt * np.pi
    beta = alpha ** 2
    w0 = (1.0 - 2.0 * beta) * np.exp(-beta)

    # 1-based wavelet 생성
    if nt is not None:
        if nt < len(w0):
            raise ValueError("nt is smaller than condition!")
        w = np.zeros(nt)  # dummy 포함
        w[0:len(w0)] = w0
    else:
        w = np.zeros(len(w0))
        w[0:] = w0

    # 1-based time axis 생성
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
    #nzbc, nxbc = vel.shape[0],vel.shape[1]
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
c1 = (-2.5)
c2 = (4.0 / 3.0)
c3 = (-1.0 / 12.0)

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

def prep_run(velocity, i_source):

    v = cp.pad(velocity.data, ((nbc, nbc), (nbc, nbc)), mode='edge')
    abc = velocity.min_vel*damp

    alpha = (v * (dt / dx)) ** 2    
    kappa = abc * dt
    temp1 = 2 + 2 * c1 * alpha - kappa
    temp2 = 1 - kappa

    bdt = (cp.asnumpy(v[isz_list[i_source], isx_list[i_source]])*dt)**2
    s_mod = bdt*s

    return temp1,temp2,alpha,s_mod


def prep_run_diff(velocity,velocity_diff, min_vel_diff, i_source):
    # velocity_diff: Nx70x70
    # min_vel_diff: N
    
    v = cp.pad(velocity.data, ((nbc, nbc), (nbc, nbc)), mode='edge')
    v_diff = cp.pad(velocity_diff, ((0,0),(nbc, nbc), (nbc, nbc)), mode='edge')
    abc_diff = min_vel_diff[:,None,None]*damp[None,:,:]

    alpha_diff = v_diff * v * (2*(dt / dx) **2)
    kappa_diff = abc_diff * dt
    temp1_diff = 2 * c1 * alpha_diff - kappa_diff
    temp2_diff = - kappa_diff

    bdt_diff = cp.asnumpy(v[None,isz_list[i_source], isx_list[i_source]]*v_diff[:,isz_list[i_source], isx_list[i_source]]) * (2*dt**2)
    s_mod_diff = bdt_diff[:,None]*s[None,:]

    return temp1_diff,temp2_diff,alpha_diff,s_mod_diff
 
@kgs.profile_each_line
def vel_to_seis(velocity, seismogram, vel_diff_vector=cp.empty((4901,0))):
    # vel_diff_vector: Nx4901
    # outputs seis_diff_vector: Nx349650
    velocity.check_constraints()     
    seismogram.check_constraints()
    
    seis_combined = []    

    vel_diff_vector = cp.transpose(vel_diff_vector)
    N = vel_diff_vector.shape[0]
    assert(vel_diff_vector.shape == (N,4901))
    do_gradient = N>0

    seis_diff_vector_combined = []
    temp1,temp2,alpha,s_mod=prep_run(velocity,0)
    if do_gradient:
        temp1_diff,temp2_diff,alpha_diff,s_mod_diff=prep_run_diff(velocity, cp.reshape(vel_diff_vector[:,:-1], (N,70,70)), vel_diff_vector[:,-1], 0)
        
    

    temp1_flat= temp1.ravel()
    temp2_flat= temp2.ravel()
    alpha_flat= alpha.ravel()
    lapg_store = cp.zeros_like(temp1)
    lapg_store_flat = lapg_store.ravel()
    if do_gradient:
        lapg_store2 = cp.zeros( (N, temp1.shape[0], temp1.shape[1]), dtype=kgs.base_type_gpu )
        lapg_store2_flat = lapg_store2.ravel()

    tx, ty = 16, 16
    bx = (nx + tx - 1) // tx
    by = (nz + ty - 1) // ty  

    

    for i_source in range(5):        
        #p_complete = cp.array(p_complete_list[i_source])
        src_idx = src_idx_list[i_source]
        _,_,_,s_mod=prep_run(velocity,i_source)
        if do_gradient:
            _,_,_,s_mod_diff=prep_run_diff(velocity, cp.reshape(vel_diff_vector[:,:-1], (N,70,70)), vel_diff_vector[:,-1], i_source)
            s_mod_diff = cp.array(s_mod_diff)

            p0_diff = cp.zeros_like(alpha_diff)
            p1_diff = cp.zeros_like(p0_diff)
            p_diff = cp.zeros_like(p0_diff)
    
            seis_diff = cp.zeros( (N,999,70) , dtype = kgs.base_type_gpu)

        p_complete = cp.zeros((nt+2,temp1.shape[0],temp1.shape[1]), dtype=kgs.base_type_gpu)
        p_complete_flat = p_complete.ravel()

        seis_diff_list = []
        for it in range(0, nt):

            if reference_mode:
                p1 = p_complete[it+1,...]
                p0 = p_complete[it,...]
                lapg_store = (cp.array(c2) * (cp.roll(p1, 1, axis=1) + cp.roll(p1, -1, axis=1) +
                               cp.roll(p1, 1, axis=0) + cp.roll(p1, -1, axis=0)) +
                         cp.array(c3) * (cp.roll(p1, 2, axis=1) + cp.roll(p1, -2, axis=1) +
                               cp.roll(p1, 2, axis=0) + cp.roll(p1, -2, axis=0)))
                p_complete[it+2,...] = (temp1 * p1 - temp2 * p0 +
                     alpha * lapg_store)
                p_complete[it+2,...].ravel()[src_idx] += s_mod[it]

            else:

                update_p(
                        (bx, by), (tx, ty),
                        (
                            temp1_flat, temp2_flat, alpha_flat,
                            p_complete_flat,
                            lapg_store_flat,
                            (it)*(nx*nz),
                            (it+1)*(nx*nz),
                            (it+2)*(nx*nz),
                            nx, nz,
                            c2, c3,
                            src_idx, s_mod[it]
                        )
                    )

            if do_gradient:
                p1 = p_complete[it+1,...]
                p0 = p_complete[it,...]
    
                #for ii in range(N):
                compute_lapg_per_slice(
                    (bx, by), (tx, ty),
                    (                        
                        p1_diff.ravel(),
                        lapg_store2_flat,
                        nx, nz, N,
                        c2, c3
                    )
                )
                
    
                p_diff = (temp1_diff*p1 + temp1*p1_diff - temp2_diff*p0 - temp2*p0_diff + 
                    alpha_diff * (
                         lapg_store
                         #cp.array(c2) * (cp.roll(p1, 1, axis=1) + cp.roll(p1, -1, axis=1) +
                         #      cp.roll(p1, 1, axis=0) + cp.roll(p1, -1, axis=0)) +
                         #cp.array(c3) * (cp.roll(p1, 2, axis=1) + cp.roll(p1, -2, axis=1) +
                         #      cp.roll(p1, 2, axis=0) + cp.roll(p1, -2, axis=0))
                     ) + 
                    alpha * (
                        lapg_store2
                         #cp.array(c2) * (cp.roll(p1_diff, 1, axis=2) + cp.roll(p1_diff, -1, axis=2) +
                         #      cp.roll(p1_diff, 1, axis=1) + cp.roll(p1_diff, -1, axis=1)) +
                         #cp.array(c3) * (cp.roll(p1_diff, 2, axis=2) + cp.roll(p1_diff, -2, axis=2) +
                         #      cp.roll(p1_diff, 2, axis=1) + cp.roll(p1_diff, -2, axis=1))
                     ))     
                #for ii in range(N):
                #    p_diff[ii,:].ravel()[src_idx] += s_mod_diff[ii,it]
                flat = p_diff.reshape(N, -1)
                # add each scalar s_mod_diff[ii,it] to all the flat[:,src_idx] positions
                flat[:, src_idx] += s_mod_diff[:, it]
    
                seis_diff[:,it,:] = p_diff[:,igz,igx]
    
                p0_diff,p1_diff,p_diff = p1_diff,p_diff,p0_diff

        seis = p_complete[2:,igz,igx]
        seis_combined.append(seis)
        if do_gradient:
            seis_diff_vector = cp.reshape(seis_diff, (N,-1))
            seis_diff_vector_combined.append(seis_diff_vector)

    if do_gradient:
        seis_diff_vector = cp.concatenate(seis_diff_vector_combined,axis=1)
    else:
        seis_diff_vector = cp.empty((0,349650),dtype=kgs.base_type_gpu)
    assert seis_diff_vector.shape == (N,349650)
    seismogram = copy.deepcopy(seismogram)
    seismogram.data = cp.stack(seis_combined)
    seismogram.check_constraints()
    jacobian = cp.transpose(seis_diff_vector)
    return seismogram,jacobian


def vel_to_seis_ref(vec):
    assert vec.shape == (4901,1)
    #assert adjoint_vec.shape == (5*999*70,1)
    v=cp.reshape(vec[:-1,0], (70,70))
    min_vel = vec[-1,0]
    
    v = cp.pad(v, ((nbc, nbc), (nbc, nbc)), mode='edge')
    abc = min_vel*damp
    v2 = v**2

    alpha = v2* (dt / dx)**2
    kappa = abc * dt
    temp1 = 2 + 2 * c1 * alpha - kappa
    temp2 = 1 - kappa

    seis_combined = cp.empty((5,999,70),dtype=kgs.base_type_gpu)

    for i_source in range(5):        
        src_idx = src_idx_list[i_source]
        bdt = (cp.asnumpy(v[isz_list[i_source], isx_list[i_source]])*dt)**2
        s_mod = bdt*s

        p_complete = cp.zeros((nt+2,temp1.shape[0],temp1.shape[1]), dtype=kgs.base_type_gpu)
        
        for it in range(0, nt):

            p1 = p_complete[it+1,...]
            p0 = p_complete[it,...]
            lapg_store = (cp.array(c2) * (cp.roll(p1, 1, axis=1) + cp.roll(p1, -1, axis=1) +
                           cp.roll(p1, 1, axis=0) + cp.roll(p1, -1, axis=0)) +
                     cp.array(c3) * (cp.roll(p1, 2, axis=1) + cp.roll(p1, -2, axis=1) +
                           cp.roll(p1, 2, axis=0) + cp.roll(p1, -2, axis=0)))
            p_complete[it+2,...] = (temp1 * p1 - temp2 * p0 +
                 alpha * lapg_store)
            p_complete[it+2,...].ravel()[src_idx] += s_mod[it]   

        seis_combined[i_source,...] = p_complete[2:,igz,igx]

    seis_combined = cp.stack(seis_combined)
    assert seis_combined.shape == (5,999,70)
    results =  seis_combined.flatten()[:,None]

    # Now create code to generate results_adjoint, such that:
    # results_adjoint == J.T * adjoint_vec
    # With J the Jacobian of vel_to_seis_ref at location vec

    return results
                

# def vel_to_seis_J(velocity):
#     sub_list = np.array_split(np.arange(4901), 49)
#     J = cp.zeros( (4901, 349650), dtype = cp.float32)
#     t=time.time()
#     for inds in sub_list:
#         print(inds[0], time.time()-t)
#         J[inds,:] = vel_to_seis_diff(velocity, cp.eye(4901)[inds,:]).astype(cp.float32)

#     return cp.transpose(J)

def vel_to_seis_J_file(velocity, filename):
    sub_list = np.array_split(np.arange(4901), 49)    
    t=time.time()
    for i,inds in enumerate(sub_list):
        print(inds[0], time.time()-t)
        J_part = cp.asnumpy(cp.ascontiguousarray(cp.transpose(vel_to_seis_diff(velocity, cp.eye(4901)[inds,:]))))
        kgs.dill_save(filename + '_64_' + str(i) + '.pickle', (J_part,inds))
        kgs.dill_save(filename + '_32_' + str(i) + '.pickle', (J_part.astype(np.float32),inds))

def vel_to_seis_J_load_file(filename, to_cpu=True):
    files = glob.glob(filename + '*')
    data = kgs.dill_load(files[0])
    if to_cpu:
        J = np.empty( (349650,4901), dtype=data[0].dtype )
    else:
        if data[0].dtype==np.float32:
            J = cp.empty( (349650, 4901), dtype=cp.float32 )
        else:
            J = cp.empty( (349650, 4901), dtype=cp.float64 )
    inds_seen = []
    for f in files:
        data = kgs.dill_load(f)
        inds = data[1]
        mat = data[0]
        if not to_cpu:
            mat = cp.array(mat)
        J[:,inds] = mat
        inds_seen.append(inds)
    inds_seen = np.sort(np.concatenate(inds_seen))
    assert np.all(inds_seen == np.arange(4901))
    return J
        
    
    sub_list = np.array_split(np.arange(4901), 49)    
    t=time.time()
    for i,inds in enumerate(sub_list):
        print(inds[0], time.time()-t)
        J_part = cp.asnumpy(cp.ascontiguousarray(cp.transpose(vel_to_seis_diff(velocity, cp.eye(4901)[inds,:]))))
        kgs.dill_save(filename + '_64_' + str(i) + '.pickle', (J_part,inds))
        kgs.dill_save(filename + '_32_' + str(i) + '.pickle', (J_part.astype(np.float32),inds))