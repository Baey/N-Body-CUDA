#include <iostream>
#include <vector>
#include "nBodyUtils.cuh"

inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess)
	{
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}


// Basic Parallel N-Body Simulation
__global__ void simulationStep(double3* p, double3* v, double* m, double3* f, size_t numBodies, double dt) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= numBodies) return;

    double3 force = make_double3(0.0, 0.0, 0.0);
    for (size_t j = 0; j < numBodies; j++) {
        if (i != j) {
            double3 diff = make_double3(
                p[j].x - p[i].x,
                p[j].y - p[i].y,
                p[j].z - p[i].z
                );
            double distSqr = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + 1e-10;
            double dist = sqrt(distSqr);
            if (dist > 0) {
                double F = G * m[i] * m[j] / distSqr;
                force.x += F * diff.x / dist;
                force.y += F * diff.y / dist;
                force.z += F * diff.z / dist;
            }
        }
    }
    f[i] = force;

    v[i].x += dt * f[i].x / m[i];
    v[i].y += dt * f[i].y / m[i];
    v[i].z += dt * f[i].z / m[i];
    p[i].x += dt * v[i].x;
    p[i].y += dt * v[i].y;
    p[i].z += dt * v[i].z;
}

// Sequential CPU N-Body Simulation
void simulationStepCPU(double3* p, double3* v, double* m, double3* f, size_t numBodies, double dt) {
    for (size_t i = 0; i < numBodies; i++) {
        double3 force = make_double3(0.0, 0.0, 0.0);
        for (size_t j = 0; j < numBodies; j++) {
            if (i != j) {
                double3 diff = make_double3(
                    p[j].x - p[i].x,
                    p[j].y - p[i].y,
                    p[j].z - p[i].z
                );
                double distSqr = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + 1e-10;
                double dist = sqrt(distSqr);
                if (dist > 0) {
                    double F = G * m[i] * m[j] / distSqr;
                    force.x += F * diff.x / dist;
                    force.y += F * diff.y / dist;
                    force.z += F * diff.z / dist;
                }
            }
        }
        f[i] = force;

        v[i].x += dt * f[i].x / m[i];
        v[i].y += dt * f[i].y / m[i];
        v[i].z += dt * f[i].z / m[i];
        p[i].x += dt * v[i].x;
        p[i].y += dt * v[i].y;
        p[i].z += dt * v[i].z;
    }
}

int main() {
    const char* filename = "solar_system.txt";
    const double dt = 0.1;
    const size_t numSteps = 1000000;
    
    size_t numBodies = getNumBodies(filename);
    double3* h_p = new double3[numBodies];
    double3* h_v = new double3[numBodies];
    double* h_m = new double[numBodies];
    loadBodiesFromFile(filename, h_p, h_v, h_m);
    printSimlulationSummary(numBodies, dt, numSteps, G, 256, 256);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Move data to GPU
    double3* d_p, *d_v, *d_f;
    double* d_m;

    checkCuda(cudaMalloc(&d_p, numBodies * sizeof(double3)));
    checkCuda(cudaMalloc(&d_v, numBodies * sizeof(double3)));
    checkCuda(cudaMalloc(&d_m, numBodies * sizeof(double)));
    checkCuda(cudaMalloc(&d_f, numBodies * sizeof(double3)));

    checkCuda(cudaMemcpy(d_p, h_p, numBodies * sizeof(double3), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_v, h_v, numBodies * sizeof(double3), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_m, h_m, numBodies * sizeof(double), cudaMemcpyHostToDevice));

    // Record the start event
    cudaEventRecord(start);

    // Run CUDA kernel
    int blockSize = 256;
    int numBlocks = std::ceil(numBodies / (float)blockSize);
    size_t step = 0;
    while(step < numSteps) {
        simulationStep<<<numBlocks, blockSize>>>(d_p, d_v, d_m, d_f, numBodies, dt);
        step++;
    }

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "\n -> GPU simulation time: " << milliseconds << " ms\n" << std::endl;

    // Retrieve data from GPU
    checkCuda(cudaMemcpy(h_p, d_p, numBodies * sizeof(double3), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(h_v, d_v, numBodies * sizeof(double3), cudaMemcpyDeviceToHost));

    // CPU Simulation
    double3* h_pCPU = new double3[numBodies];
    double3* h_vCPU = new double3[numBodies];
    double* h_mCPU = new double[numBodies];
    double3* h_fCPU = new double3[numBodies];
    loadBodiesFromFile(filename, h_pCPU, h_vCPU, h_mCPU);

    step = 0;
    while(step < numSteps) {
        simulationStepCPU(h_pCPU, h_vCPU, h_mCPU, h_fCPU, numBodies, dt);
        step++;
    }

    // Free memory
    cudaFree(d_p);
    cudaFree(d_v);
    cudaFree(d_m);
    cudaFree(d_f);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    // Compare results
    for (int i = 0; i < numBodies; i++) {
        double pos_error = sqrt((h_p[i].x - h_pCPU[i].x) * (h_p[i].x - h_pCPU[i].x) +
                                (h_p[i].y - h_pCPU[i].y) * (h_p[i].y - h_pCPU[i].y) +
                                (h_p[i].z - h_pCPU[i].z) * (h_p[i].z - h_pCPU[i].z));
        double vel_error = sqrt((h_v[i].x - h_vCPU[i].x) * (h_v[i].x - h_vCPU[i].x) +
                                (h_v[i].y - h_vCPU[i].y) * (h_v[i].y - h_vCPU[i].y) +
                                (h_v[i].z - h_vCPU[i].z) * (h_v[i].z - h_vCPU[i].z));
        std::cout << "Body " << i << " Position error: " << pos_error << " Velocity error: " << vel_error << std::endl;
    }


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
