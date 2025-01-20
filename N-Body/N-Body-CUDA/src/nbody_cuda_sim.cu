#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>

#include <cuda_runtime.h>
#include "kernel.cuh"
#include "cuda_utils.cuh"
#include "nbody_utils.cuh"



int main(int argc, char* argv[]){
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <initial_states_file>" << " <output_file>" << " <dt>" << " <num_steps>" << " <block_size>"  << std::endl;
        return 1;
    }
    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    const double dt = std::stod(argv[3]);
    const size_t numSteps = std::stoul(argv[4]);
    const size_t block_size = std::stoul(argv[5]);

    if (block_size <= 0) {
        std::cerr << "Invalid block size value: " << block_size << std::endl;
        return 1;
    }
    if (dt <= 0) {
        std::cerr << "Invalid time step value: " << dt << std::endl;
        return 1;
    }
    if (numSteps <= 0) {
        std::cerr << "Invalid number of steps value: " << numSteps << std::endl;
        return 1;
    }
    
    size_t numBodies = getNumBodies(inputFile);
    if (numBodies <= 0) {
        std::cerr << "Invalid number of bodies value: " << numBodies << std::endl;
        return 1;
    }
    double3* h_p = new double3[numBodies];
    double3* h_v = new double3[numBodies];
    double* h_m = new double[numBodies];
    loadBodiesFromFile(inputFile, h_p, h_v, h_m);

    dim3 block(block_size);
    dim3 grid((numBodies + block_size - 1) / block_size);
    printSimulationSummary(numBodies, dt, numSteps, G, block.x, grid.x);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Move data to GPU
    double3 *d_p, *d_v, *d_a, *d_f;
    double *d_m;

    checkCuda(cudaMalloc(&d_p, numBodies * sizeof(double3)));
    checkCuda(cudaMalloc(&d_v, numBodies * sizeof(double3)));
    checkCuda(cudaMalloc(&d_a, numBodies * sizeof(double3)));
    checkCuda(cudaMalloc(&d_m, numBodies * sizeof(double)));
    checkCuda(cudaMalloc(&d_f, numBodies * numBodies * sizeof(double3)));

    checkCuda(cudaMemcpy(d_p, h_p, numBodies * sizeof(double3), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_v, h_v, numBodies * sizeof(double3), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_m, h_m, numBodies * sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(d_f, 0, numBodies * numBodies * sizeof(double3)));
    checkCuda(cudaMemset(d_a, 0, numBodies * sizeof(double3)));

    // Record the start event
    cudaEventRecord(start);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::ofstream outFile(outputFile, std::ios::binary);
    size_t step = 0;
    while(step < numSteps) {
        nBodyVelocityVerlet<<<grid, block>>>(d_p, d_v, d_a, d_m, d_f, numBodies, dt);
        cudaMemcpyAsync(h_p, d_p, numBodies * sizeof(double3), cudaMemcpyDeviceToHost, stream);
        for (size_t i = 0; i < numBodies; ++i) {
            outFile << std::fixed << std::setprecision(10) << h_p[i].x << " " << h_p[i].y << " " << h_p[i].z << " ";
        }
        outFile << std::endl;
        step++;
    }

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    outFile.close();
    cudaStreamDestroy(stream);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "\n -> GPU simulation time: " << milliseconds << " ms\n" << std::endl;

    // Free memory
    cudaFree(d_p);
    cudaFree(d_v);
    cudaFree(d_m);
    cudaFree(d_f);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    // Free CPU memory
    delete[] h_p;
    delete[] h_v;
    delete[] h_m;

    return 0;
}