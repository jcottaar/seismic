# Original source: XXX
# Python translation adapted from: https://www.kaggle.com/code/jaewook704/waveform-inversion-vel-to-seis by Jae-Wook Kim

import numpy as np
import cupy as cp
import kaggle_support as kgs
import copy

# CUDA kernel to update p and add source
kernel_code = r'''
extern "C" __global__
void update_p(
              const double* __restrict__ temp1,
              const double* __restrict__ temp2,
              const double* __restrict__ alpha,
              double*             __restrict__ pout,
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
    float t1 = pout[ind_offset2 + iy  * nx + ix_p1]
             + pout[ind_offset2 + iy  * nx + ix_m1]
             + pout[ind_offset2 + iy_p1 * nx + ix  ]
             + pout[ind_offset2 + iy_m1 * nx + ix  ];
    // Collect neighbors (±2)
    float t2 = pout[ind_offset2 + iy  * nx + ix_p2]
             + pout[ind_offset2 + iy  * nx + ix_m2]
             + pout[ind_offset2 + iy_p2 * nx + ix  ]
             + pout[ind_offset2 + iy_m2 * nx + ix  ];

    float out = temp1[idx]*pout[ind_offset2 + idx] 
              - temp2[idx]*pout[ind_offset1+idx]
              + alpha[idx]*(c2*t1 + c3*t2);

    // fused source injection:
    if (idx == src_idx) {
        out += s_val;
    }
    pout[ind_offset3+idx] = out;
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
        isx = int(round(coord['sx'] / dx)) + nbc
        isz = int(round(coord['sz'] / dx)) + nbc
        igx = (np.round(np.array(coord['gx']) / dx) + nbc).astype(int)
        igz = (np.round(np.array(coord['gz']) / dx) + nbc).astype(int)
    
        if abs(coord['sz']) < 0.5:
            isz += 1
        igz = igz + (np.abs(np.array(coord['gz'])) < 0.5).astype(int)
        return isx, isz, igx, igz

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

    def padvel(v0, nbc):
        v_padded = cp.pad(v0, ((nbc, nbc), (nbc, nbc)), mode='edge')
        return v_padded


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

    damp = AbcCoef2D(310,310, nbc, dx)

    v = padvel(velocity.data, nbc)
    abc = velocity.min_vel*damp

    alpha = (v * (dt / dx)) ** 2
    nx,nz = alpha.shape
    kappa = abc * dt
    temp1 = 2 + 2 * c1 * alpha - kappa
    temp2 = 1 - kappa
    beta_dt = (v * dt) ** 2
    s = expand_source(s, nt)
    isx, isz, igx, igz = adjust_sr(coord, dx, nbc)

    src_idx = np.int32(isz*nx + isx)

    bdt = cp.asnumpy(beta_dt[isz, isx])
    s_mod = bdt*s
    
    s_mod = s_mod.astype(kgs.base_type)

    return temp1,temp2,nx,nz,nt,c2,c3,alpha,src_idx,s_mod,igz,igx



@kgs.profile_each_line
def vel_to_seis(velocity,seismogram):
    velocity.check_constraints()
    seismogram.check_constraints()
    seis_combined = []
    for i_source in range(5):
        temp1,temp2,nx,nz,nt,c2,c3,alpha,src_idx,s_mod,igz,igx=prep_run(velocity,i_source)

        temp1_flat= temp1.ravel()
        temp2_flat= temp2.ravel()
        alpha_flat= alpha.ravel()

        p_complete = cp.zeros((nt+2,temp1.shape[0],temp1.shape[1]), dtype=kgs.base_type_gpu)
        p_complete_flat = p_complete.ravel()
    
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
                    temp1_flat, temp2_flat, alpha_flat,
                    p_complete_flat,
                    (it)*(nx*nz),
                    (it+1)*(nx*nz),
                    (it+2)*(nx*nz),
                    nx, nz,
                    c2, c3,
                    src_idx, s_mod[it]
                )
            )

        seis = p_complete[2:,igz,igx]
        seis_combined.append(seis)

    seismogram = copy.deepcopy(seismogram)
    seismogram.data = cp.stack(seis_combined)
    seismogram.check_constraints()

    return seismogram
            
