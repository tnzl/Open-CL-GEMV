#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <CL/cl.hpp>
#include <chrono>  // For measuring time

#define INPUT_SIZE 2048
#define WEIGHT_SIZE 2048
#define NUM_ITERATIONS 10000

// Function to initialize OpenCL and run VM kernel
void runVM() {
    // Initialize OpenCL
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms.front();
    
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices.front();
    
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
    
    // Load and build the kernel
    cl::Program::Sources sources;
    std::string kernel_code;
    std::ifstream kernelFile("GEMV.cl");
    if (!kernelFile.is_open()) {
        std::cerr << "Failed to open kernel file!" << std::endl;
        return;
    }
    
    std::stringstream ss;
    ss << kernelFile.rdbuf();
    kernel_code = ss.str();
    kernelFile.close();
    
    sources.push_back({kernel_code.c_str(), kernel_code.length()});
    
    cl::Program program(context, sources);
    if (program.build({device}) != CL_SUCCESS) {
        std::cerr << "Error building kernel: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return;
    }
    
    // Create memory buffers
    std::vector<uint8_t> input(INPUT_SIZE);  // Vector (size INPUT_SIZE)
    std::vector<uint8_t> weight(INPUT_SIZE * WEIGHT_SIZE);  // Matrix (size INPUT_SIZE x WEIGHT_SIZE)
    std::vector<uint32_t> output(WEIGHT_SIZE);  // Output vector (size WEIGHT_SIZE)
    
    // Initialize input and weight matrices
    // (In a real scenario, initialize these with meaningful values)
    for (int i = 0; i < INPUT_SIZE; ++i) {
        input[i] = 1; //static_cast<uint8_t>(i % 256); // Example: Initialize input to values from 0 to 255
        for (int j = 0; j < WEIGHT_SIZE; ++j) {
            weight[i * WEIGHT_SIZE + j] = j; //static_cast<uint8_t>((i + j) % 256); // Example: Initialize weight to values from 0 to 255
        }
    }
    
    // Create OpenCL buffers
    cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint8_t) * INPUT_SIZE, input.data());
    cl::Buffer weightBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint8_t) * INPUT_SIZE * WEIGHT_SIZE, weight.data());
    cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uint32_t) * WEIGHT_SIZE);
    
    // Set up kernel and arguments
    cl::Kernel kernel(program, "vm");  // Using "vm" as the kernel function name for vector-matrix multiplication
    kernel.setArg(0, inputBuffer);
    kernel.setArg(1, weightBuffer);
    kernel.setArg(2, outputBuffer);
    kernel.setArg(3, INPUT_SIZE);
    kernel.setArg(4, WEIGHT_SIZE);
    
    double totalElapsedTime = 0.0;

    // Measure kernel execution time over multiple iterations
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(WEIGHT_SIZE), cl::NullRange, nullptr, &event);
        event.wait();
        
        cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        double elapsedTime = static_cast<double>(endTime - startTime);
        totalElapsedTime += elapsedTime;
    }

    // Calculate average time
    double averageTime = totalElapsedTime / NUM_ITERATIONS;

    std::cout << "Average kernel execution time over " << NUM_ITERATIONS << " iterations: " << averageTime << " ns" << std::endl;
    
    // Read the result from the device (optional)
    // queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(uint32_t) * WEIGHT_SIZE, output.data());
    // Display output (optional)
    // for (int i = 0; i < WEIGHT_SIZE; ++i) {
    //     std::cout << "Output[" << i << "]: " << output[i] << std::endl;
    // }
}

int main() {
    runVM();
    return 0;
}
