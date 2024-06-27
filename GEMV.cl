// Kernel to perform VM (vector-matrix multiplication) operation
__kernel void vm(__global uchar* input, __global uchar* weight, __global uint* output, int inputSize, int weightSize) {
    int gid = get_global_id(0);
    uint sum = 0;
    
    if (gid < weightSize) {
        for (int i = 0; i < inputSize; ++i) {
            sum += input[i] * weight[i * weightSize + gid];
        }
        output[gid] = sum;
    }
}
