extern "C" __global__
__launch_bounds__(256) void k_amax_reduce(const float* x, float* amax) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_co = _gid;
        float acc0 = -1e+30f;
        for (int a0 = a0_co; a0 < 1043; a0 += 256) {
            float in0 = x[a0];
            acc0 = fmaxf(acc0, in0);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 16);
            acc0 = fmaxf(acc0, acc0__o);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 8);
            acc0 = fmaxf(acc0, acc0__o);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 4);
            acc0 = fmaxf(acc0, acc0__o);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 2);
            acc0 = fmaxf(acc0, acc0__o);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 1);
            acc0 = fmaxf(acc0, acc0__o);
        }
        __shared__ float acc0_smem[8];
        if (lane == 0) {
            acc0_smem[warp] = acc0;
        }
        __syncthreads();
        for (int s = 4; s > 0; s >>= 1) {
            if (warp < s) {
                float acc0 = acc0_smem[warp];
                float acc0__o = acc0_smem[warp + s];
                acc0 = fmaxf(acc0, acc0__o);
                acc0_smem[warp] = acc0;
            }
            __syncthreads();
        }
        acc0 = acc0_smem[0];
        if (a0_co == 0) {
            amax[0] = acc0;
        }
}
