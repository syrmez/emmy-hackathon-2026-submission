extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const float* x0, const float* x1, float* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 257) {
        int a0 = _gid;
        float in0 = x0[a0];
        float in1 = x1[a0];
        float v0 = in0 + in1;
        add[a0] = v0;
    }
}
