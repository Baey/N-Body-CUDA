#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
    return result;
}

#endif