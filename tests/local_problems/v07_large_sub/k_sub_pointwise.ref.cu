extern "C" __global__
__launch_bounds__(256) void k_sub_pointwise(const float* x0, const float* x1, float* sub) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 2731) {
        int a0 = _gid;
        float in0__u0 = x0[a0 * 3];
        float in1__u0 = x1[a0 * 3];
        float v0__u0 = in0__u0 - in1__u0;
        float in0__u1 = x0[a0 * 3 + 1];
        float in1__u1 = x1[a0 * 3 + 1];
        float v0__u1 = in0__u1 - in1__u1;
        float in0__u2 = x0[a0 * 3 + 2];
        float in1__u2 = x1[a0 * 3 + 2];
        float v0__u2 = in0__u2 - in1__u2;
        sub[a0 * 3] = v0__u0;
        sub[a0 * 3 + 1] = v0__u1;
        sub[a0 * 3 + 2] = v0__u2;
    }
}
