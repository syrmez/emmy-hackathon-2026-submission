extern "C" __global__
__launch_bounds__(256) void k_div_pointwise(const float* add_c1, const float* x, float* div) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 171) {
        int a0 = _gid;
        float in1__u0 = x[a0 * 3];
        float in0__u0 = add_c1[0];
        float v3__u0 = in1__u0 / sqrtf((in0__u0 + (in1__u0 * in1__u0)));
        float in1__u1 = x[a0 * 3 + 1];
        float in0__u1 = add_c1[0];
        float v3__u1 = in1__u1 / sqrtf((in0__u1 + (in1__u1 * in1__u1)));
        float in1__u2 = x[a0 * 3 + 2];
        float in0__u2 = add_c1[0];
        float v3__u2 = in1__u2 / sqrtf((in0__u2 + (in1__u2 * in1__u2)));
        div[a0 * 3] = v3__u0;
        div[a0 * 3 + 1] = v3__u1;
        div[a0 * 3 + 2] = v3__u2;
    }
}
