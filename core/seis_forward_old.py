# Original source: XXX
# Python translation adapted from: https://www.kaggle.com/code/jaewook704/waveform-inversion-vel-to-seis by Jae-Wook Kim

import numpy as np
import cupy as cp
import kaggle_support as kgs
import copy

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
        w = np.zeros(nt + 1)  # dummy 포함
        w[1:len(w0) + 1] = w0
    else:
        w = np.zeros(len(w0) + 1)
        w[1:] = w0

    # 1-based time axis 생성
    if nt is not None:
        tw = np.arange(1, len(w)) * dt
    else:
        tw = np.arange(1, len(w)) * dt

    return w, tw

def padvel(v0, nbc):
    v_padded = np.pad(v0, ((nbc, nbc), (nbc, nbc)), mode='edge')
    nz, nx = v_padded.shape
    v = np.zeros((nz + 1, nx + 1))
    v[1:, 1:] = v_padded
    return v
	
def expand_source(s0, nt):
    s0 = np.asarray(s0).flatten()
    s = np.zeros(nt + 1)
    s[1:len(s0) + 1] = s0
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

@kgs.profile_each_line
def a2d_mod_abc24(v, min_vel, nbc, dx, nt, dt, s, coord):
    ng = len(coord['gx'])
    seis = np.zeros((nt + 1, ng))  # 1-based time axis

    c1 = -2.5
    c2 = 4.0 / 3.0
    c3 = -1.0 / 12.0

    v = padvel(v, nbc)
    abc = AbcCoef2D(v, min_vel, nbc, dx)

    alpha = (v * dt / dx) ** 2
    kappa = abc * dt
    temp1 = 2 + 2 * c1 * alpha - kappa
    temp2 = 1 - kappa
    beta_dt = (v * dt) ** 2
    s = expand_source(s, nt)
    isx, isz, igx, igz = adjust_sr(coord, dx, nbc)

    p0 = np.zeros_like(v)
    p1 = np.zeros_like(v)

    dtype = cp.float32

    temp1=cp.array(temp1, dtype=dtype)
    temp2=cp.array(temp2, dtype=dtype)
    p0=cp.array(p0, dtype=dtype)
    p1=cp.array(p1, dtype=dtype)
    p=cp.array(p1, dtype=dtype)
    #c2=cp.array(c2, dtype=dtype)
    #c3=cp.array(c3, dtype=dtype)
    alpha=cp.array(alpha, dtype=dtype)
    #beta_dt = cp.array(beta_dt, dtype=dtype)
    #s = cp.array(s, dtype=dtype)
    seis = cp.array(seis, dtype=dtype)
    igz = cp.array(igz)
    igx = cp.array(igx)

    # Flattened strides for the kernel—only cheap views, no new allocs
    # Note: ravel() here does *not* copy
    p0_flat   = p0.ravel()
    p1_flat   = p1.ravel()
    temp1_flat= temp1.ravel()
    temp2_flat= temp2.ravel()
    alpha_flat= alpha.ravel()
    p_flat = p.ravel()

    nx,ny = alpha.shape
    tx, ty = 16, 16
    bx = (nx + tx - 1) // tx
    by = (ny + ty - 1) // ty  

    # Cast your scalars once
    nx_i32 = np.int32(nx)
    ny_i32 = np.int32(ny)
    c2_f32  = np.float32(c2)
    c3_f32  = np.float32(c3)

    recv_idx = (igz*nx + igx).astype(np.int32)
    src_idx = np.int32(isz*nx + isx)

    bdt = beta_dt[isz, isx]

    s_mod = np.float32(bdt*s)

    # Time Loop (1-based)
    for it in range(1, nt + 1):
        # p2 = (temp1 * p1 - temp2 * p0 +
        #      alpha * (
        #          c2 * (cp.roll(p1, 1, axis=1) + cp.roll(p1, -1, axis=1) +
        #                cp.roll(p1, 1, axis=0) + cp.roll(p1, -1, axis=0)) +
        #          c3 * (cp.roll(p1, 2, axis=1) + cp.roll(p1, -2, axis=1) +
        #                cp.roll(p1, 2, axis=0) + cp.roll(p1, -2, axis=0))
        #      ))

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
    
def vel_to_seis(vel, min_vel):

    # 1. 모델 및 파라미터 설정
    nz = 70
    nx = 70
    dx = 10
    nbc = 120
    nt = 1000
    dt = 1e-3
    freq = 15
    isFS = False  # 자유표면 사용 여부
    
    # 2. Ricker 파형 생성
    s, _ = ricker(freq, dt)
    
    # 3. 소스 및 수신기 설정
    coord = {}
    coord['sz'] = 1 * dx
    coord['gx'] = np.arange(0, nx) * dx
    coord['gz'] = np.ones_like(coord['gx']) * dx
    
    # 4. 파동장 시뮬레이션 수행 : 소스 위치만 바꿔서
    seis_data = []
    for source_x in [0, 17, 34, 52, 69]:
        coord['sx'] = source_x * dx        
        
        # 시뮬레이션
        seis = a2d_mod_abc24(vel, min_vel, nbc, dx, nt, dt, s, coord)

        seis_data += [seis]
        
    return np.stack(seis_data, axis=0)