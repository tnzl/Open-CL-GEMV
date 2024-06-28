#include <vector>

// Function to calculate gemv 
template<typename T1, typename T2>
void gemv(const std::vector<T1>& input, const std::vector<T1>& weight, std::vector<T2>& output, int inputSize, int weightSize) {
    for (int j = 0; j < weightSize; ++j) {
        T2 sum = 0;
        for (int i = 0; i < inputSize; ++i) {
            sum += input[i] * weight[i * weightSize + j];
        }
        output[j] = sum;
    }
}

// Function to calculate L2 norm of difference and index of highest difference
template<typename T1, typename T2>
void l2_norm(const std::vector<T1>& outputGPU, const std::vector<T2>& outputCPU, double& l2norm, double& maxDiffIndex) {
    l2norm = 0.0;
    double maxDiff = 0.0;
    maxDiffIndex = -1;

    for (size_t i = 0; i < outputGPU.size(); ++i) {
        double diff = std::fabs(static_cast<double>(outputGPU[i]) - static_cast<double>(outputCPU[i]));
        l2norm += diff * diff; // Accumulate squared differences for L2 norm

        if (diff > maxDiff) {
            maxDiff = diff;
            maxDiffIndex = static_cast<int>(i);
        }
    }

    l2norm = std::sqrt(l2norm); // Compute square root to get L2 norm
}

// Function to calculate transposed matrix
template <typename T>
void transposeMatrix(const std::vector<T>& matrix, std::vector<T>& transposedMatrix, size_t height, size_t width) {
    transposedMatrix.resize(width * height);
    for (size_t row = 0; row < height; row++) {
        for (size_t col = 0; col < width; col++) {
            transposedMatrix[col * height + row] = matrix[row * width + col];
        }
    }
}

// Function to quantize float to uint8_t
uint8_t quantize(float value, float scale, uint8_t zero_point) {
    return static_cast<uint8_t>(std::round(value / scale) + zero_point);
}

// Function to dequantize uint32_t to float
float dequantize(uint32_t value, float scale, uint32_t zero_point) {
    return static_cast<float>(value - zero_point) * scale;
}
