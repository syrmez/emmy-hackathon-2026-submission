extern "C" __global__
__launch_bounds__(8) void k_div_reduce(const float* x, float* div) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 8;
        int a1_co = (_gid) % 8;
        float acc0 = 0.0f;
        for (int a1 = a1_co; a1 < 97; a1 += 8) {
            float in0 = x[a0 * 97 + a1];
            float v0 = in0 * in0;
            acc0 += v0;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 4);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 2);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 1);
            acc0 = acc0 + acc0__o;
        }
        float v2 = 1.0f / sqrtf(acc0);
        for (int a2 = a1_co; a2 < 97; a2 += 8) {
            float in1 = x[a0 * 97 + a2];
            float v3 = in1 * v2;
            div[a0 * 97 + a2] = v3;
        }
}
