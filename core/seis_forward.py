# Original source: XXX
# Python translation adapted from: https://www.kaggle.com/code/jaewook704/waveform-inversion-vel-to-seis by Jae-Wook Kim

import numpy as np
import cupy as cp
import kaggle_support as kgs
import copy

base_type = np.float32
base_type_gpu = cp.float32

# CUDA kernel to update p and add source
kernel_code = r'''
extern "C" __global__
void update_p(const float* __restrict__ p0,
              const float* __restrict__ p1,
              const float* __restrict__ temp1,
              const float* __restrict__ temp2,
              const float* __restrict__ alpha,
              float*             __restrict__ pout,
              const int    nx,
              const int    ny,
              const float  c2,
              const float  c3,
              const int   src_idx,
              const float s_val) {
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
    float t1 = p1[iy  * nx + ix_p1]
             + p1[iy  * nx + ix_m1]
             + p1[iy_p1 * nx + ix  ]
             + p1[iy_m1 * nx + ix  ];
    // Collect neighbors (±2)
    float t2 = p1[iy  * nx + ix_p2]
             + p1[iy  * nx + ix_m2]
             + p1[iy_p2 * nx + ix  ]
             + p1[iy_m2 * nx + ix  ];

    float out = temp1[idx]*p1[idx] 
              - temp2[idx]*p0[idx]
              + alpha[idx]*(c2*t1 + c3*t2);

    // fused source injection:
    if (idx == src_idx) {
        out += s_val;
    }
    pout[idx] = out;
}
'''
module = cp.RawModule(code=kernel_code)
update_p = module.get_function('update_p')




def prep_run(velocity, i_source):
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
    	
    def adjust_sr(coord, dx, nbc):
        isx = int(round(coord['sx'] / dx)) + 1 + nbc
        isz = int(round(coord['sz'] / dx)) + 1 + nbc
        igx = (np.round(np.array(coord['gx']) / dx) + 1 + nbc).astype(int)
        igz = (np.round(np.array(coord['gz']) / dx) + 1 + nbc).astype(int)
    
        if abs(coord['sz']) < 0.5:
            isz += 1
        igz = igz + (np.abs(np.array(coord['gz'])) < 0.5).astype(int)
        return isx, isz, igx, igz
    	
    def AbcCoef2D(vel, velmin, nbc, dx):
        nzbc, nxbc = vel.shape[1] - 1, vel.shape[0] - 1  # 실제 사이즈
        velmin = np.min(vel[1:, 1:])
        nz = nzbc - 2 * nbc
        nx = nxbc - 2 * nbc
    
        a = (nbc - 1) * dx
        kappa = 3.0 * velmin * np.log(1e7) / (2.0 * a)
    
        damp1d = kappa * (((np.arange(1, nbc + 1) - 1) * dx / a) ** 2)
        damp = np.zeros((nzbc + 1, nxbc + 1))
    
        for iz in range(1, nzbc + 1):
            damp[iz, 1:nbc + 1] = damp1d[::-1]
            damp[iz, nx + nbc + 1 : nx + 2 * nbc + 1] = damp1d
    
        for ix in range(nbc + 1, nbc + nx + 1):
            damp[1:nbc + 1, ix] = damp1d[::-1]
            damp[nz + nbc + 1 : nz + 2 * nbc + 1, ix] = damp1d
    
        return damp

    def padvel(v0, nbc):
        v_padded = np.pad(v0, ((nbc, nbc), (nbc, nbc)), mode='edge')
        nz, nx = v_padded.shape
        v = np.zeros((nz + 1, nx + 1))
        v[1:, 1:] = v_padded
        return v	


    nz = 70
    nx = 70
    dx = 10
    nbc = 120
    nt = 999
    dt = (1e-3)
    freq = 15
    s, _ = (ricker(freq, dt))

    c1 = (-2.5)
    c2 = (4.0 / 3.0)
    c3 = (-1.0 / 12.0)

    coord = {}
    source_x = [0, 17, 34, 52, 69][i_source]
    coord['sx'] = source_x * dx        
    coord['sz'] = 1 * dx
    coord['gx'] = np.arange(0, nx) * dx
    coord['gz'] = np.ones_like(coord['gx']) * dx

    ng = len(coord['gx'])

    v = padvel(velocity.data, nbc)
    abc = AbcCoef2D(v, velocity.min_vel, nbc, dx)

    alpha = (v * dt / dx) ** 2
    nx,nz = alpha.shape
    kappa = abc * dt
    temp1 = 2 + 2 * c1 * alpha - kappa
    temp2 = 1 - kappa
    beta_dt = (v * dt) ** 2
    s = expand_source(s, nt)
    isx, isz, igx, igz = adjust_sr(coord, dx, nbc)
    seis = np.zeros((nt + 1, ng))

    p0 = np.zeros_like(v)
    p1 = np.zeros_like(v)
    p = np.zeros_like(v)

    src_idx = np.int32(isz*nx + isx)

    bdt = beta_dt[isz, isx]
    s_mod = bdt*s
    recv_idx = (igz*nx + igx).astype(np.int32)

    seis = np.zeros((nt, ng))
    
    
   

    p0 = cp.array(p0, dtype=base_type_gpu)
    p1 = cp.array(p1, dtype=base_type_gpu)
    p = cp.array(p, dtype=base_type_gpu)
    temp1 = cp.array(temp1, dtype=base_type_gpu)
    temp2 = cp.array(temp2, dtype=base_type_gpu)
    nx = np.int32(nx)
    nz = np.int32(nz)
    nt = np.int32(nt)
    c2 = np.array(c2).astype(base_type)
    c3 = np.array(c3).astype(base_type)
    alpha = cp.array(alpha, dtype=base_type_gpu)
    src_idx = np.int32(src_idx)
    s_mod = s_mod.astype(base_type)
    recv_idx = cp.array(np.int32(recv_idx))
    seis = cp.array(seis, dtype=base_type_gpu)

    return p0,p1,p,temp1,temp2,nx,nz,nt,c2,c3,alpha,src_idx,s_mod,recv_idx,seis

@kgs.profile_each_line
def vel_to_seis(velocity,seismogram):
    velocity.check_constraints()
    seis_combined = []
    for i_source in range(5):
        p0,p1,p,temp1,temp2,nx,nz,nt,c2,c3,alpha,src_idx,s_mod,recv_idx,seis=prep_run(velocity,i_source)

        #print(recv_idx,seis)

        p0_flat   = p0.ravel()
        p1_flat   = p1.ravel()
        temp1_flat= temp1.ravel()
        temp2_flat= temp2.ravel()
        alpha_flat= alpha.ravel()
        p_flat = p.ravel()
    
        tx, ty = 16, 16
        bx = (nx + tx - 1) // tx
        by = (nz + ty - 1) // ty  

        for it in range(0, nt):
            # p = (temp1 * p1 - temp2 * p0 +
            #      alpha * (
            #          cp.array(c2) * (cp.roll(p1, 1, axis=1) + cp.roll(p1, -1, axis=1) +
            #                cp.roll(p1, 1, axis=0) + cp.roll(p1, -1, axis=0)) +
            #          cp.array(c3) * (cp.roll(p1, 2, axis=1) + cp.roll(p1, -2, axis=1) +
            #                cp.roll(p1, 2, axis=0) + cp.roll(p1, -2, axis=0))
            #      ))
            # p[isz, isx] +=  beta_dt[isz, isx] * s[it]
            update_p(
                (bx, by), (tx, ty),
                (
                    p0_flat, p1_flat,
                    temp1_flat, temp2_flat, alpha_flat,
                    p_flat,
                    nx, nz,
                    c2, c3,
                    src_idx, s_mod[it]
                )
            )

            cp.take(p_flat, recv_idx, out=seis[it])
    
            p0, p1, p = p1, p, p0
            p0_flat, p1_flat, p_flat = p1_flat, p_flat, p0_flat

        seis_combined.append(cp.asnumpy(seis))

    seismogram = copy.deepcopy(seismogram)
    seismogram.data = np.stack(seis_combined)
    seismogram.check_constraints()

    return seismogram
            



@kgs.profile_each_line
def a2d_mod_abc24(v, min_vel, nbc, dx, nt, dt, s, coord):
    
    # Flattened strides for the kernel—only cheap views, no new allocs
    # Note: ravel() here does *not* copy
    p0_flat   = p0.ravel()
    p1_flat   = p1.ravel()
    temp1_flat= temp1.ravel()
    temp2_flat= temp2.ravel()
    alpha_flat= alpha.ravel()
    p_flat = p.ravel()

    tx, ty = 16, 16
    bx = (nx + tx - 1) // tx
    by = (ny + ty - 1) // ty  

    # Cast your scalars once
    nx_i32 = np.int32(nx)
    ny_i32 = np.int32(ny)
    c2_f32  = np.float32(c2)
    c3_f32  = np.float32(c3)

    
    

    

    # Time Loop (1-based)
    for it in range(1, nt + 1):
        # p2 = (temp1 * p1 - temp2 * p0 +
        #      alpha * (
        #          c2 * (cp.roll(p1, 1, axis=1) + cp.roll(p1, -1, axis=1) +
        #                cp.roll(p1, 1, axis=0) + cp.roll(p1, -1, axis=0)) +
        #          c3 * (cp.roll(p1, 2, axis=1) + cp.roll(p1, -2, axis=1) +
        #                cp.roll(p1, 2, axis=0) + cp.roll(p1, -2, axis=0))
        #      ))

        print(p_flat.dtype, s_mod.dtype, src_idx.dtype)
        update_p(
            (bx, by), (tx, ty),
            (
                p0_flat, p1_flat,
                temp1_flat, temp2_flat, alpha_flat,
                p_flat,
                nx_i32, ny_i32,
                c2_f32, c3_f32,
                src_idx, s_mod[it]
            )
        )

        #p=pout
        # Source
        #p[isz, isx] +=  beta_dt[isz, isx] * s[it]

        #for ig in range(ng):
        #seis[it, :] = p[igz[0], igx]
        #seis[it] = p_flat[recv_idx]
        cp.take(p_flat, recv_idx, out=seis[it])
        #print(p[recv_idx].shape)

        p0, p1, p = p1, p, p0
        p0_flat, p1_flat, p_flat = p1_flat, p_flat, p0_flat

    seis = cp.asnumpy(seis)
    return seis
    
# def vel_to_seis_old(vel, min_vel):

    
#     # 3. 소스 및 수신기 설정
    
    
#     # 4. 파동장 시뮬레이션 수행 : 소스 위치만 바꿔서
#     seis_data = []
    
        
#         # 시뮬레이션
#         seis = a2d_mod_abc24(vel, min_vel, nbc, dx, nt, dt, s, coord)

#         seis_data += [seis]
        
#     return np.stack(seis_data, axis=0)