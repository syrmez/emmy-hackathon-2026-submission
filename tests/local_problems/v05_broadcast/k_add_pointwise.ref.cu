extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const float* x1, const float* x0, float* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 481) {
        int a0 = _gid / 37;
        int a1 = (_gid) % 37;
        float in0 = x0[a0 * 37 + a1];
        float in1 = x1[a1];
        float v0 = in0 + in1;
        add[a0 * 37 + a1] = v0;
    }
}
