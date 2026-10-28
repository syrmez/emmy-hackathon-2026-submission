extern "C" __global__
__launch_bounds__(256) void k_relu_pointwise(const float* sub_c1, const float* mul_c1, const float* x, float* relu) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1031) {
        int a0 = _gid;
        float in0 = mul_c1[0];
        float in2 = x[a0];
        float in1 = sub_c1[0];
        float v2 = fmaxf(0.0f, ((in0 * in2) - in1));
        relu[a0] = v2;
    }
}
