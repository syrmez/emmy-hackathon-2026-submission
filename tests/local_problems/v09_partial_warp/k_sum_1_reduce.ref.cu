extern "C" __global__
__launch_bounds__(8) void k_sum_1_reduce(const float* x, float* sum_1) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_co = _gid;
        float acc0 = 0.0f;
        for (int a0 = a0_co; a0 < 7; a0 += 8) {
            float in0 = x[a0];
            acc0 += in0;
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
        if (a0_co == 0) {
            sum_1[0] = acc0;
        }
}
