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
        std::cerr << "Usage: " << argv[0] << " <num_bodies>" << " <dt>" << " <num_steps>" << " <block_size>" << " <method>" << std::endl;
        return 1;
    }
    size_t numBodies = std::stoul(argv[1]);
    const double dt = std::stod(argv[2]);
    const size_t numSteps = std::stoul(argv[3]);
    const size_t block_size = std::stoul(argv[4]);
    const unsigned int method = std::stoul(argv[5]);

    if (method != 0 && method != 1 && method != 2) {
        std::cerr << "Invalid method value: " << method << " (0/1/2)" << std::endl;
        return 1;
    }
    if (block_size <= 0) {
        std::cerr << "Invalid block size value: " << block_size << std::endl;
        return 1;
    }
    if (numBodies <= 0) {
        std::cerr << "Invalid number of bodies value: " << numBodies << std::endl;
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

    dim3 block(block_size);
    dim3 grid((numBodies + block_size - 1) / block_size);
    if (method == 0) {
        std::cout << "Using standard N-Body simulation method" << std::endl;
    } else if (method == 1)
    {
        std::cout << "Using standard N-Body simulation method with velocity Verlet solver" << std::endl;
    } else {
        std::cout << "Using optimized N-Body simulation method" << std::endl;
        int numElements = numBodies * (numBodies - 1) / 2;
        int numBlocks = (numElements + block_size - 1) / block_size;
        grid = dim3(numBlocks);
    }

    
    double3* h_p = new double3[numBodies];
    double3* h_v = new double3[numBodies];
    double* h_m = new double[numBodies];

    printSimulationSummary(numBodies, dt, numSteps, G, block.x, grid.x);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Move data to GPU
    double3 *d_p, *d_v, *d_a, *d_f, *d_diff;
    double *d_m, *d_distSqr, *d_dist;

    checkCuda(cudaMalloc(&d_p, numBodies * sizeof(double3)));
    checkCuda(cudaMalloc(&d_v, numBodies * sizeof(double3)));
    checkCuda(cudaMalloc(&d_a, numBodies * sizeof(double3)));
    checkCuda(cudaMalloc(&d_m, numBodies * sizeof(double)));
    checkCuda(cudaMalloc(&d_f, numBodies * numBodies * sizeof(double3)));
    checkCuda(cudaMalloc(&d_diff, numBodies * numBodies * sizeof(double3)));
    checkCuda(cudaMalloc(&d_distSqr, numBodies * numBodies * sizeof(double)));
    checkCuda(cudaMalloc(&d_dist, numBodies * numBodies * sizeof(double)));

    checkCuda(cudaMemcpy(d_p, h_p, numBodies * sizeof(double3), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_v, h_v, numBodies * sizeof(double3), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_m, h_m, numBodies * sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(d_f, 0, numBodies * numBodies * sizeof(double3)));
    checkCuda(cudaMemset(d_a, 0, numBodies * sizeof(double3)));
    checkCuda(cudaMemset(d_diff, 0, numBodies * numBodies * sizeof(double3)));
    checkCuda(cudaMemset(d_distSqr, 0, numBodies * numBodies * sizeof(double)));

    // Record the start event
    std::cout << "Starting simulation with method " << method << "..." << std::endl;
    cudaEventRecord(start);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    size_t step = 0;

    if (method == 0) {
        while(step < numSteps) {
            nBody<<<grid, block>>>(d_p, d_v, d_m, d_f, numBodies, dt);
            step++;
        }
    } else if (method == 1) {
        while(step < numSteps) {
            nBodyVelocityVerlet<<<grid, block>>>(d_p, d_v, d_a, d_m, d_f, numBodies, dt);
            step++;
        }
    } else {
        while(step < numSteps) {
            computeDist<<<grid, block, block_size>>>(d_p, d_diff, d_distSqr, d_dist, numBodies);
            computeForce<<<grid, block, block_size>>>(d_m, d_diff, d_distSqr, d_dist, d_f, numBodies);
            updateBody<<<grid, block, block_size>>>(d_p, d_v, d_f, d_m, numBodies, dt);
            step++;
        }
    }

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaStreamDestroy(stream);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "\n -> GPU simulation time: " << milliseconds << " ms\n" << std::endl;

    // CPU Simulation
    double3* h_pCPU = new double3[numBodies];
    double3* h_vCPU = new double3[numBodies];
    double3* h_aCPU = new double3[numBodies];
    double* h_mCPU = new double[numBodies];
    double3* h_fCPU = new double3[numBodies];
    // loadBodiesFromFile(inputFile, h_pCPU, h_vCPU, h_mCPU);

    auto startCPU = std::chrono::high_resolution_clock::now();
    step = 0;
    if (method == 0) {
        while(step < numSteps) {
            nbodyStepCPU(h_pCPU, h_vCPU, h_mCPU, h_fCPU, numBodies, dt);
            step++;
        }
    } else {
        while(step < numSteps) {
            nbodyStepVelocityVerletCPU(h_pCPU, h_vCPU, h_aCPU, h_mCPU, h_fCPU, numBodies, dt);
            step++;
        }
    }

    auto stopCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = stopCPU - startCPU;
    std::cout << "\n -> CPU simulation time: " << cpuDuration.count() << " ms\n" << std::endl;

    // Free memory
    cudaFree(d_p);
    cudaFree(d_v);
    cudaFree(d_m);
    cudaFree(d_f);
    cudaFree(d_diff);
    cudaFree(d_distSqr);
    cudaFree(d_dist);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free CPU memory
    delete[] h_p;
    delete[] h_v;
    delete[] h_m;
    delete[] h_pCPU;
    delete[] h_vCPU;
    delete[] h_mCPU;
    delete[] h_fCPU;

    return 0;
}