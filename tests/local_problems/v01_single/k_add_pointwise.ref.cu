extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const float* x0, const float* x1, float* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        float in0 = x0[0];
        float in1 = x1[0];
        float v0 = in0 + in1;
        add[0] = v0;
    }
}
