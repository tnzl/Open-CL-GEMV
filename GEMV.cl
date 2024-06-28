__kernel void vm(__global uchar* input, __global uchar* weight, __global uint* output, int inputSize, int weightSize) {
    int gid = get_global_id(0);
    __local uchar localInput[256]; // Example size, tune according to your device capabilities

    uint sum = 0;
    
    for (int i = 0; i < inputSize; i += 256) {
        // Load chunk of input into local memory
        int localId = get_local_id(0);
        if (localId < 256 && i + localId < inputSize) {
            localInput[localId] = input[i + localId];
        }
        // printf("Global ID: %d, Local ID: %d\n", gid, localId);
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Perform multiplication for this chunk
        if (gid < weightSize) {
            for (int j = 0; j < 256 && i + j < inputSize; ++j) {
                sum += localInput[j] * weight[(i + j) * weightSize + gid];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (gid < weightSize) {
        output[gid] = sum;
    }
}
