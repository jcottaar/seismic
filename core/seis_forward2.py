# Original source: XXX
# Python translation adapted from: https://www.kaggle.com/code/jaewook704/waveform-inversion-vel-to-seis by Jae-Wook Kim

import numpy as np
import cupy as cp
import kaggle_support as kgs
import copy
import seis_numerics
import time

N_source_to_do = 5

profile_vals = dict()
profiling = False
profile_time = 0
def reset_profile():
    global sync_vals
    profile_vals = dict()

def profile(name):
    if profiling:
        cp.cuda.Stream.null.synchronize()
        global profile_time
        global profile_vals
        time_diff = time.time()-profile_time
        profile_time = time.time()
        if not name in profile_vals:
            profile_vals[name] = []
        profile_vals[name].append(time_diff)

def show_profile(N_run):
    for key,value in profile_vals.items():
        if not key=='start':
            print(f"{key}: {np.sum(value)/N_run:.4f}")
        
stream = cp.cuda.Stream(non_blocking=True)
graph = 0
graph_diff = 0
graph_adjoint = 0

@kgs.profile_each_line
def vel_to_seis(vec, vec_diff=None, vec_adjoint=None, adjoint_on_residual=False):
    # Outputs:
    # result: the seismogram associated with velocity field vec
    # result_diff: J@vec_diff, where J is the Jacobian of the operation above
    # result_adjoint: J^T@vec_adjoint, or J^T@(result-vec_adjoint) if adjoint_on_residual=True
    profile('start')
    assert vec.shape == (4901,1)
    assert vec_adjoint is None or vec_adjoint.shape == (5*999*70,1)
    assert vec_diff is None or vec_diff.shape == (4901,1)
    do_diff = not (vec_diff is None)
    do_adjoint = not (vec_adjoint is None)

    profile('init')

    # PREPARATION
    prep_run(vec)
     
    if do_diff:
        prep_run_diff(vec_diff)        

    if do_adjoint:        
        #alpha_adjoint = cp.zeros_like(alpha)
        #temp1_adjoint = cp.zeros_like(alpha)
        #temp2_adjoint = cp.zeros_like(alpha)
        #v_adjoint = cp.zeros_like(v)  
        alpha_adjoint[...] = 0
        temp1_adjoint[...] = 0
        temp2_adjoint[...] = 0
        v_adjoint[...] = 0
        seis_combined_adjoint[...] = cp.reshape(vec_adjoint, (5,999,70))

    tx, ty = 32, 32
    bx = (nx + tx - 1) // tx
    by = (nz + ty - 1) // ty  

    

    # LOOP

    global stream
    global graph
    global graph_adjoint

    cp.cuda.Stream.null.synchronize()

    with stream:

        profile('prep for source loop')
        
        for i_source in range(N_source_to_do):     
            src_idx = src_idx_list[i_source]
            bdt = (v[isz_list[i_source], isx_list[i_source]]*dt)**2
            s_mod[...] = bdt*s
    
            src_idx_dev[...] = cp.array(src_idx, dtype=cp.int32)
    
            profile('prep for time loop')

            if graph==0:
                stream.begin_capture()
                print('capturing graph')
                for it in range(0, nt):
        
                    update_p(
                                (bx, by), (tx, ty),
                                (
                                    temp1_flat, temp2_flat, alpha_flat,
                                    p_complete_flat[it*(nx*nz):],
                                    p_complete_flat[(it+1)*(nx*nz):],
                                    p_complete_flat[(it+2)*(nx*nz):],
                                    lapg_store_flat[nx*nz*it:],
                                    nx, nz, it,
                                    c2, c3,
                                    src_idx_dev, s_mod
                                )
                            )
                graph = stream.end_capture()
                graph.upload(stream)
                profile('stream capture')

            graph.launch(stream)
            stream.synchronize()
    
            profile('time loop')
            seis_combined[i_source,...] = p_complete[2:,igz_dev,igx_dev]
    
            profile('extract seis')
    
            if do_diff:                
                bdt_diff = 2*((v[isz_list[i_source], isx_list[i_source]]*v_diff[isz_list[i_source], isx_list[i_source]]))* dt**2
                s_mod_diff[...] = bdt_diff*s

                for it in range(0, nt):
                    lapg_store_diff[...] = (cp.array(c2) * (cp.roll( p_complete_diff[it+1,...], 1, axis=1) + cp.roll( p_complete_diff[it+1,...], -1, axis=1) +
                                   cp.roll( p_complete_diff[it+1,...], 1, axis=0) + cp.roll( p_complete_diff[it+1,...], -1, axis=0)) +
                             cp.array(c3) * (cp.roll( p_complete_diff[it+1,...], 2, axis=1) + cp.roll( p_complete_diff[it+1,...], -2, axis=1) +
                                   cp.roll( p_complete_diff[it+1,...], 2, axis=0) + cp.roll( p_complete_diff[it+1,...], -2, axis=0)))
                    p_complete_diff[it+2,...] = (temp1 * p_complete_diff[it+1,...] + temp1_diff*p_complete[it+1,...] - temp2_diff * p_complete[it,...] - temp2*p_complete_diff[it,...] + 
                         alpha_diff * lapg_store[it,...] + alpha*lapg_store_diff)
                    p_complete_diff[it+2,...].ravel()[src_idx] += s_mod_diff[it]   

                seis_combined_diff[i_source,...] = p_complete_diff[2:,igz_dev,igx_dev]

                profile('diff')
                
            if do_adjoint:     
                #p_complete_adjoint =  cp.zeros((nt+2,temp1.shape[0],temp1.shape[1]), dtype=kgs.base_type_gpu)
                p_complete_adjoint[...] = 0
                if adjoint_on_residual:
                    p_complete_adjoint[2:,igz_dev,igx_dev] = seis_combined[i_source,...]-seis_combined_adjoint[i_source,...]
                else:
                    p_complete_adjoint[2:,igz_dev,igx_dev] = seis_combined_adjoint[i_source,...]

                #p_complete_adjoint[...] = 0
                s_mod_adjoint[...] = 0
                #s_mod_adjoint = cp.zeros_like(s_mod)
                #s_mod_adjoint_flat = s_mod_adjoint.ravel(); p_complete_adjoint_flat = p_complete_adjoint.ravel();
                #temp1_adjoint_flat = temp1_adjoint.ravel();temp2_adjoint_flat = temp2_adjoint.ravel();
                #alpha_adjoint_flat = alpha_adjoint.ravel();
                profile('prep for time loop adjoint')

                if graph_adjoint==0:
                    stream.begin_capture()
                    print('capturing graph')
                    
                    
                    for it in np.arange(nt-1,-1,-1):       
                        
                        update_p_adjoint(
                                (bx, by), (tx, ty),
                                (
                                    temp1_flat, temp2_flat, alpha_flat,
                                    p_complete_flat,
                                    lapg_store_flat,
                                    s_mod_adjoint, p_complete_adjoint_flat, temp1_adjoint_flat, temp2_adjoint_flat, alpha_adjoint_flat,
                                    (it)*(nx*nz),
                                    (it+1)*(nx*nz),
                                    (it+2)*(nx*nz),
                                    nx, nz, it,
                                    c2, c3,
                                    src_idx_dev
                                )
                            )

                    graph_adjoint = stream.end_capture()
                    graph_adjoint.upload(stream)       

                #p_complete_adjoint[...] = 0
                

                graph_adjoint.launch(stream)
                # for it in np.arange(nt-1,-1,-1):       
                        
                #     update_p_adjoint(
                #             (bx, by), (tx, ty),
                #             (
                #                 temp1_flat, temp2_flat, alpha_flat,
                #                 p_complete_flat,
                #                 lapg_store_flat,
                #                 s_mod_adjoint, p_complete_adjoint_flat, temp1_adjoint_flat, temp2_adjoint_flat, alpha_adjoint_flat,
                #                 (it)*(nx*nz),
                #                 (it+1)*(nx*nz),
                #                 (it+2)*(nx*nz),
                #                 nx, nz, it,
                #                 c2, c3,
                #                 src_idx_dev
                #             )
                #             )
                stream.synchronize()
                profile('time loop adjoint')
        
                #bdt_adjoint[...] = cp.sum(s_mod_adjoint*cp.array(s))
                v_adjoint[isz_list[i_source], isx_list[i_source]] += 2*dt**2 * v[isz_list[i_source], isx_list[i_source]] * cp.sum(s_mod_adjoint*s)
                #del p_complete_adjoint
    
                profile('end adjoint')
                
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

    profile('finish')

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

    #return v,temp1,temp2,alpha


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


def vel_to_seis_ref(vec, vec_diff=None, vec_adjoint=None, adjoint_on_residual=False):
    # Outputs:
    # result: the seismogram associated with velocity field vec
    # result_diff: J@vec_diff, where J is the Jacobian of the operation above
    # result_adjoint: J^T@vec_adjoint, or J^T@(result-vec_adjoint) if adjoint_on_residual=True
    assert vec.shape == (4901,1)
    assert vec_adjoint is None or vec_adjoint.shape == (5*999*70,1)
    assert vec_diff is None or vec_diff.shape == (4901,1)
    do_diff = not (vec_diff is None)
    do_adjoint = not (vec_adjoint is None)

    v,temp1,temp2,alpha = prep_run(vec)

    seis_combined = cp.zeros((5,999,70),dtype=kgs.base_type_gpu)
    p_complete_list = []

    for i_source in range(N_source_to_do):     
        src_idx = src_idx_list[i_source]
        bdt = (cp.asnumpy(v[isz_list[i_source], isx_list[i_source]])*dt)**2
        s_mod = bdt*s

        p_complete = cp.zeros((nt+2,temp1.shape[0],temp1.shape[1]), dtype=kgs.base_type_gpu)
        lapg_store = cp.zeros_like(p_complete)
        
        for it in range(0, nt):

            p1 = p_complete[it+1,...]
            p0 = p_complete[it,...]
            lapg_store[it+2,...] = (cp.array(c2) * (cp.roll(p1, 1, axis=1) + cp.roll(p1, -1, axis=1) +
                           cp.roll(p1, 1, axis=0) + cp.roll(p1, -1, axis=0)) +
                     cp.array(c3) * (cp.roll(p1, 2, axis=1) + cp.roll(p1, -2, axis=1) +
                           cp.roll(p1, 2, axis=0) + cp.roll(p1, -2, axis=0)))
            p_complete[it+2,...] = (temp1 * p1 - temp2 * p0 +
                 alpha * lapg_store[it+2,...])
            p_complete[it+2,...].ravel()[src_idx] += s_mod[it]   

        seis_combined[i_source,...] = p_complete[2:,igz,igx]
        p_complete_list.append((cp.asnumpy(p_complete), cp.asnumpy(lapg_store)))

    assert seis_combined.shape == (5,999,70)
    result =  seis_combined.flatten()[:,None]

    # DIFF

    if do_diff:

        v_diff,temp1_diff,temp2_diff,alpha_diff = prep_run_diff(vec_diff,v)
        
        seis_combined_diff = cp.zeros((5,999,70),dtype=kgs.base_type_gpu)
        for i_source in range(N_source_to_do):
            src_idx = src_idx_list[i_source]
            bdt_diff = 2*((cp.asnumpy(v[isz_list[i_source], isx_list[i_source]]*v_diff[isz_list[i_source], isx_list[i_source]])))* dt**2
            s_mod_diff = bdt_diff*s
            
            (p_complete, lapg_store) = p_complete_list[i_source]
            p_complete = cp.array(p_complete)
            lapg_store = cp.array(lapg_store)
            p_complete_diff = cp.zeros((nt+2,temp1.shape[0],temp1.shape[1]), dtype=kgs.base_type_gpu)
    
            for it in range(0, nt):
                p1_diff = p_complete_diff[it+1,...]
                p0_diff = p_complete_diff[it,...]
                p1 = p_complete[it+1,...]
                p0 = p_complete[it,...]
                lapg_store_diff = (cp.array(c2) * (cp.roll(p1_diff, 1, axis=1) + cp.roll(p1_diff, -1, axis=1) +
                               cp.roll(p1_diff, 1, axis=0) + cp.roll(p1_diff, -1, axis=0)) +
                         cp.array(c3) * (cp.roll(p1_diff, 2, axis=1) + cp.roll(p1_diff, -2, axis=1) +
                               cp.roll(p1_diff, 2, axis=0) + cp.roll(p1_diff, -2, axis=0)))
                p_complete_diff[it+2,...] = (temp1 * p1_diff + temp1_diff*p1 - temp2_diff * p0 - temp2*p0_diff + 
                     alpha_diff * lapg_store[it+2,...] + alpha*lapg_store_diff)
                p_complete_diff[it+2,...].ravel()[src_idx] += s_mod_diff[it]   
    
            seis_combined_diff[i_source,...] = p_complete_diff[2:,igz,igx]
    
        assert seis_combined_diff.shape == (5,999,70)
        result_diff =  seis_combined_diff.flatten()[:,None]
    else:
        result_diff = None

        

    # ADJOINT

    if do_adjoint:
        if adjoint_on_residual:
            vec_adjoint = result-vec_adjoint
        seis_combined_adjoint = cp.reshape(vec_adjoint, (5,999,70))
        alpha_adjoint = cp.zeros_like(alpha)
        temp1_adjoint = cp.zeros_like(alpha)
        temp2_adjoint = cp.zeros_like(alpha)
        v_adjoint = cp.zeros_like(v)
        for i_source in range(N_source_to_do):
            src_idx = src_idx_list[i_source]
            
            (p_complete, lapg_store) = p_complete_list[i_source]
            p_complete = cp.array(p_complete)
            lapg_store = cp.array(lapg_store)
    
            p_complete_adjoint =  cp.zeros((nt+2,temp1.shape[0],temp1.shape[1]), dtype=kgs.base_type_gpu)
            p_complete_adjoint[2:,igz,igx] = seis_combined_adjoint[i_source,...]
    
            s_mod_adjoint = cp.zeros_like(s_mod)
            for it in np.arange(nt-1,-1,-1):
                
                s_mod_adjoint[it] = p_complete_adjoint[it+2,...].ravel()[src_idx]
                p_complete_adjoint[it+1,...] += temp1 * p_complete_adjoint[it+2,...]
                temp1_adjoint+=p_complete[it+1,...] * p_complete_adjoint[it+2,...]
                p_complete_adjoint[it,...] -= temp2 * p_complete_adjoint[it+2,...]
                temp2_adjoint-=p_complete[it,...] * p_complete_adjoint[it+2,...]
                alpha_adjoint+=lapg_store[it+2,...] * p_complete_adjoint[it+2,...]
                lapg_store_adjoint = alpha*p_complete_adjoint[it+2,...]
                p1_adjoint = (cp.array(c2) * (cp.roll(lapg_store_adjoint, -1, axis=1) + cp.roll(lapg_store_adjoint, 1, axis=1) +
                                  cp.roll(lapg_store_adjoint, -1, axis=0) + cp.roll(lapg_store_adjoint, 1, axis=0)) +
                  cp.array(c3) * (cp.roll(lapg_store_adjoint, -2, axis=1) + cp.roll(lapg_store_adjoint, 2, axis=1) +
                                  cp.roll(lapg_store_adjoint, -2, axis=0) + cp.roll(lapg_store_adjoint, 2, axis=0)))
                p_complete_adjoint[it+1,...]+= p1_adjoint
                
    
            bdt_adjoint = cp.sum(s_mod_adjoint*cp.array(s))
            v_adjoint[isz_list[i_source], isx_list[i_source]] += 2*dt**2 * v[isz_list[i_source], isx_list[i_source]] * bdt_adjoint
            
        result_adjoint = prep_run_adjoint(v_adjoint,temp1_adjoint,temp2_adjoint,alpha_adjoint,v)
    else:
        result_adjoint = None


    return result, result_diff, result_adjoint






# CUDA kernel to update p and add source
kernel_code = r'''
extern "C" __global__
void update_p(
              const double* __restrict__ temp1,
              const double* __restrict__ temp2,
              const double* __restrict__ alpha,
              const double*             __restrict__ pout,
              const double*             __restrict__ pout1,
              double*             __restrict__ pout2,
              double*             __restrict__ lapg_store,
              const int    nx,
              const int    ny,
              const int    it,
              const double  c2,
              const double  c3,
              const int*   __restrict__ src_idx,
              const double* __restrict__ s_mod) {
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

    double t1;
    double t2;
    
    // Collect neighbors (±1)
    t1 = pout1[iy  * nx + ix_p1]
             + pout1[iy  * nx + ix_m1]
             + pout1[iy_p1 * nx + ix  ]
             + pout1[iy_m1 * nx + ix  ];
    // Collect neighbors (±2)
    t2 = pout1[iy  * nx + ix_p2]
             + pout1[iy  * nx + ix_m2]
             + pout1[iy_p2 * nx + ix  ]
             + pout1[iy_m2 * nx + ix  ];

    double lapg_store_local = (c2*t1+c3*t2);
    lapg_store[idx] = lapg_store_local;

    double temp1_local = __ldg(&temp1[idx]);
    double temp2_local = __ldg(&temp2[idx]);
    double pout1_local = __ldg(&pout1[idx]);
    double pout_local = __ldg(&pout[idx]);
    double alpha_local = __ldg(&alpha[idx]);
    double out = temp1_local*pout1_local
              - temp2_local*pout_local
              + alpha_local*lapg_store_local;

    // fused source injection:
    if (idx == src_idx[0]) {
        out += s_mod[it];
    }
    pout2[idx] = out;
}
'''

module = cp.RawModule(code=kernel_code)
update_p = module.get_function('update_p')


# CUDA kernel for adjoint calculation
kernel_code = r'''
extern "C" __global__
void update_p_adjoint(
              const double* __restrict__ temp1,
              const double* __restrict__ temp2,
              const double* __restrict__ alpha,
              const double* __restrict__ p_complete,
              const double* __restrict__ lapg_store,
              double*  __restrict__ s_mod_adjoint,
              double*  __restrict__ p_complete_adjoint,
              double*  __restrict__ temp1_adjoint,
              double*  __restrict__ temp2_adjoint,
              double*  __restrict__ alpha_adjoint,
              const int    ind_offset1,
              const int    ind_offset2,
              const int    ind_offset3,
              const int    nx,
              const int    ny,
              const int    it,
              const double  c2,
              const double  c3,
              const int*   __restrict__ src_idx) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;
    int idx = iy * nx + ix;
    int idx1 = ind_offset1 + idx;
    int idx2 = ind_offset2 + idx;
    int idx3 = ind_offset3 + idx;

    // Manual wrap at ±1, ±2
    int ix_p1 = ix+1; if (ix_p1==nx)  ix_p1=0;
    int ix_m1 = ix-1; if (ix_m1<0)    ix_m1=nx-1;
    int ix_p2 = ix+2; if (ix_p2>=nx)  ix_p2-=nx;
    int ix_m2 = ix-2; if (ix_m2<0)     ix_m2+=nx;
    int iy_p1 = iy+1; if (iy_p1==ny)  iy_p1=0;
    int iy_m1 = iy-1; if (iy_m1<0)    iy_m1=ny-1;
    int iy_p2 = iy+2; if (iy_p2>=ny)  iy_p2-=ny;
    int iy_m2 = iy-2; if (iy_m2<0)     iy_m2+=ny;

    if (idx == src_idx[0]) {
        s_mod_adjoint[it] = p_complete_adjoint[idx3];
    }

    p_complete_adjoint[idx2] += temp1[idx]*p_complete_adjoint[idx3];
    temp1_adjoint[idx] += p_complete[idx2]*p_complete_adjoint[idx3];
    p_complete_adjoint[idx1] -= temp2[idx]*p_complete_adjoint[idx3];
    temp2_adjoint[idx] -= p_complete[idx1]*p_complete_adjoint[idx3];
    alpha_adjoint[idx] += lapg_store[idx1] * p_complete_adjoint[idx3];
    //double lapg_store_adjoint[idx] = alpha[idx] * p_complete_adjoint[idx3];

    // Collect neighbors (±1)
    double t1 = alpha[iy  * nx + ix_p1] * p_complete_adjoint[ind_offset3 + iy  * nx + ix_p1] +
                alpha[iy  * nx + ix_m1] * p_complete_adjoint[ind_offset3 + iy  * nx + ix_m1] +
                alpha[iy_p1  * nx + ix] * p_complete_adjoint[ind_offset3 + iy_p1  * nx + ix] +
                alpha[iy_m1  * nx + ix] * p_complete_adjoint[ind_offset3 + iy_m1  * nx + ix];
    // Collect neighbors (±2)
    double t2 = alpha[iy  * nx + ix_p2] * p_complete_adjoint[ind_offset3 + iy  * nx + ix_p2] +
                alpha[iy  * nx + ix_m2] * p_complete_adjoint[ind_offset3 + iy  * nx + ix_m2] +
                alpha[iy_p2  * nx + ix] * p_complete_adjoint[ind_offset3 + iy_p2  * nx + ix] +
                alpha[iy_m2  * nx + ix] * p_complete_adjoint[ind_offset3 + iy_m2  * nx + ix];

    p_complete_adjoint[idx2]+=c2*t1+c3*t2;
      
}
'''
module = cp.RawModule(code=kernel_code)
update_p_adjoint = module.get_function('update_p_adjoint')









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
s = cp.array(s, dtype=kgs.base_type_gpu)
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


seis_combined = cp.zeros((5,999,70),dtype=kgs.base_type_gpu)
p_complete = cp.zeros((nt+2,nx,nz), dtype=kgs.base_type_gpu)
lapg_store = cp.zeros((nt,nx,nz), dtype=kgs.base_type_gpu)
p_complete_flat = p_complete.ravel();lapg_store_flat=lapg_store.ravel()
temp1 = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
temp2 = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
alpha = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
v = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
temp1_flat = temp1.ravel();temp2_flat = temp2.ravel();alpha_flat = alpha.ravel()        
s_mod = cp.zeros_like(s)

seis_combined_diff = cp.zeros((5,999,70),dtype=kgs.base_type_gpu)
p_complete_diff = cp.zeros((nt+2,nx,nz), dtype=kgs.base_type_gpu)
lapg_store_diff = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
temp1_diff = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
temp2_diff = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
alpha_diff = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
v_diff = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
s_mod_diff = cp.zeros_like(s)

seis_combined_adjoint = cp.zeros((5,999,70),dtype=kgs.base_type_gpu)
p_complete_adjoint = cp.zeros((nt+2,nx,nz), dtype=kgs.base_type_gpu)
lapg_store_adjoint = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
temp1_adjoint = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
temp2_adjoint = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
alpha_adjoint = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
v_adjoint = cp.zeros((nx,nz), dtype=kgs.base_type_gpu)
s_mod_adjoint = cp.zeros_like(s)
p_complete_adjoint_flat = p_complete_adjoint.ravel()
lapg_store_adjoint_flat = lapg_store_adjoint.ravel()
temp1_adjoint_flat = temp1_adjoint.ravel()
temp2_adjoint_flat = temp2_adjoint.ravel()
alpha_adjoint_flat = alpha_adjoint.ravel()
v_adjoint_flat = v_adjoint.ravel()

src_idx_dev = cp.zeros((1,), dtype=cp.int32)


igz_dev,igx_dev = cp.array(igz), cp.array(igx)