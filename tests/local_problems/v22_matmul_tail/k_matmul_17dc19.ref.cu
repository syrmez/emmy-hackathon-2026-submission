extern "C" __global__
__launch_bounds__(1024) void k_matmul_17dc19(const float* x1, const float* x0, float* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_b = _gid / 1024;
        int a1_b = (_gid / 1024) % 1;
        int a0_u = (_gid / 64) % 16;
        int a1_u = (_gid) % 64;
        float acc0__c0_0 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c2_0 = 0.0f;
        float acc0__c3_0 = 0.0f;
        float acc0__c4_0 = 0.0f;
        float acc0__c5_0 = 0.0f;
        #pragma unroll
        for (int a2 = 0; a2 < 17; a2++) {
            float in0__ar0 = x0[(a0_b * 96 + a0_u * 6) % 13 * 17 + a2];
            float in1__bc0 = x1[a2 * 19 + (a1_b * 64 + a1_u) % 19];
            float v0__c0_0 = in0__ar0 * in1__bc0;
            acc0__c0_0 += v0__c0_0;
            float in0__ar1 = x0[(a0_b * 96 + a0_u * 6 + 1) % 13 * 17 + a2];
            float v0__c1_0 = in0__ar1 * in1__bc0;
            acc0__c1_0 += v0__c1_0;
            float in0__ar2 = x0[(a0_b * 96 + a0_u * 6 + 2) % 13 * 17 + a2];
            float v0__c2_0 = in0__ar2 * in1__bc0;
            acc0__c2_0 += v0__c2_0;
            float in0__ar3 = x0[(a0_b * 96 + a0_u * 6 + 3) % 13 * 17 + a2];
            float v0__c3_0 = in0__ar3 * in1__bc0;
            acc0__c3_0 += v0__c3_0;
            float in0__ar4 = x0[(a0_b * 96 + a0_u * 6 + 4) % 13 * 17 + a2];
            float v0__c4_0 = in0__ar4 * in1__bc0;
            acc0__c4_0 += v0__c4_0;
            float in0__ar5 = x0[(a0_b * 96 + a0_u * 6 + 5) % 13 * 17 + a2];
            float v0__c5_0 = in0__ar5 * in1__bc0;
            acc0__c5_0 += v0__c5_0;
        }
        if (a0_b * 96 + a0_u * 6 + 0 < 13 && a1_b * 64 + a1_u * 1 + 0 < 19) {
            matmul[(a0_b * 96 + a0_u * 6) % 13 * 19 + (a1_b * 64 + a1_u) % 19] = acc0__c0_0;
        }
        if (a0_b * 96 + a0_u * 6 + 1 < 13 && a1_b * 64 + a1_u * 1 + 0 < 19) {
            matmul[(a0_b * 96 + a0_u * 6 + 1) % 13 * 19 + (a1_b * 64 + a1_u) % 19] = acc0__c1_0;
        }
        if (a0_b * 96 + a0_u * 6 + 2 < 13 && a1_b * 64 + a1_u * 1 + 0 < 19) {
            matmul[(a0_b * 96 + a0_u * 6 + 2) % 13 * 19 + (a1_b * 64 + a1_u) % 19] = acc0__c2_0;
        }
        if (a0_b * 96 + a0_u * 6 + 3 < 13 && a1_b * 64 + a1_u * 1 + 0 < 19) {
            matmul[(a0_b * 96 + a0_u * 6 + 3) % 13 * 19 + (a1_b * 64 + a1_u) % 19] = acc0__c3_0;
        }
        if (a0_b * 96 + a0_u * 6 + 4 < 13 && a1_b * 64 + a1_u * 1 + 0 < 19) {
            matmul[(a0_b * 96 + a0_u * 6 + 4) % 13 * 19 + (a1_b * 64 + a1_u) % 19] = acc0__c4_0;
        }
        if (a0_b * 96 + a0_u * 6 + 5 < 13 && a1_b * 64 + a1_u * 1 + 0 < 19) {
            matmul[(a0_b * 96 + a0_u * 6 + 5) % 13 * 19 + (a1_b * 64 + a1_u) % 19] = acc0__c5_0;
        }
}
