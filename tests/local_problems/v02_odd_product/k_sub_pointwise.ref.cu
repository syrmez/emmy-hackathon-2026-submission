extern "C" __global__
__launch_bounds__(256) void k_sub_pointwise(const float* sub_c1, const float* x1, const float* x0, float* sub) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 127) {
        int a0 = _gid;
        float in1 = x0[a0];
        float in2 = x1[a0];
        float in0 = sub_c1[0];
        float v1 = (in1 * in2) - in0;
        sub[a0] = v1;
    }
}
