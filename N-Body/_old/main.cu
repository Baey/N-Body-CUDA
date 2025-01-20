#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include "nBodyUtils.cuh"

#define BLOCK_SIZE 256

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

void printMatrix(const char* name, double* d_matrix, size_t rows, size_t cols) {
    double* h_matrix = new double[rows * cols];
    cudaMemcpy(h_matrix, d_matrix, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "Matrix: " << name << "\n";
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(3)
                      << h_matrix[i * cols + j] << " ";
        }
        std::cout << "\n";
    }

    delete[] h_matrix;
}

void printMatrix3(const char* name, double3* d_matrix, size_t rows, size_t cols) {
    double3* h_matrix = new double3[rows * cols];
    cudaMemcpy(h_matrix, d_matrix, rows * cols * sizeof(double3), cudaMemcpyDeviceToHost);

    std::cout << "Matrix: " << name << "\n";
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double3 value = h_matrix[i * cols + j];
            std::cout << std::fixed << std::setprecision(3) << "("
                      << value.x << ", " << value.y << ", " << value.z << ") ";
        }
        std::cout << "\n";
    }

    delete[] h_matrix;
}


// Basic Parallel N-Body Simulation
__global__ void nBodyV1(double3* p, double3* v, double* m, double3* f, size_t numBodies, double dt) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= numBodies) return;

    double3 p_i = p[i];
    double3 v_i = v[i];
    double m_i = m[i];

    double3 force = make_double3(0.0, 0.0, 0.0);
    for (size_t j = 0; j < numBodies; j++) {
        if (i != j) {
            double3 p_j = p[j];
            double m_j = m[j];

            double3 diff = make_double3(
                p_j.x - p_i.x,
                p_j.y - p_i.y,
                p_j.z - p_i.z
            );

            double distSqr = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + 1e-10;
            double dist = sqrt(distSqr);
            if (dist > 0) {
                double F = G * m_i * m_j / distSqr;
                force.x += F * diff.x / dist;
                force.y += F * diff.y / dist;
                force.z += F * diff.z / dist;
            }
        }
    }
    f[i] = force;
    
    v_i = make_double3(
        v_i.x + dt * force.x / m_i,
        v_i.y + dt * force.y / m_i,
        v_i.z + dt * force.z / m_i
    );
    p[i] = make_double3(
        p_i.x + dt * v_i.x,
        p_i.y + dt * v_i.y,
        p_i.z + dt * v_i.z
        );
    v[i] = v_i;
}

// N-Body Simulation with Velocity Verlet Integration
__global__ void nBodyVelocityVerlet(double3* p, double3* v, double3* a, double* m, double3* f, size_t numBodies, double dt) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= numBodies) return;

    double3 p_i = p[i];
    double3 v_i = v[i];
    double3 a_i = a[i];
    double m_i = m[i];

    double3 p_new = make_double3(
        p_i.x + v_i.x * dt + 0.5 * a_i.x * dt * dt,
        p_i.y + v_i.y * dt + 0.5 * a_i.y * dt * dt,
        p_i.z + v_i.z * dt + 0.5 * a_i.z * dt * dt
    );

    double3 force = make_double3(0.0, 0.0, 0.0);
    for (size_t j = 0; j < numBodies; j++) {
        if (i != j) {
            double3 p_j = p[j];
            double m_j = m[j];

            double3 diff = make_double3(
                p_j.x - p_i.x,
                p_j.y - p_i.y,
                p_j.z - p_i.z
            );

            double distSqr = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + 1e-10;
            double dist = sqrt(distSqr);
            if (dist > 0) {
                double F = G * m_i * m_j / distSqr;
                force.x += F * diff.x / dist;
                force.y += F * diff.y / dist;
                force.z += F * diff.z / dist;
            }
        }
    }
    double3 a_new = make_double3(force.x / m_i, force.y / m_i, force.z / m_i);

    double3 v_new = make_double3(
        v_i.x + 0.5 * (a_i.x + a_new.x) * dt,
        v_i.y + 0.5 * (a_i.y + a_new.y) * dt,
        v_i.z + 0.5 * (a_i.z + a_new.z) * dt
    );

    p[i] = p_new;
    v[i] = v_new;
    a[i] = a_new;
    f[i] = force;
}

__global__ void computeDist(double3* p, double3* diff, double* distSqr, double* dist, size_t numBodies) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double3 sh_p[BLOCK_SIZE];

    int i = 0;
    int sum = numBodies - 1;
    while (idx >= sum) {
        idx -= sum;
        sum -= 1;
        i += 1;
    }
    int j = i + 1 + idx;

    if (threadIdx.x < numBodies) {
        sh_p[threadIdx.x] = p[threadIdx.x];
    }
    __syncthreads();

    if (i < numBodies && j < numBodies) {
        double3 p_i = sh_p[i];
        double3 p_j = sh_p[j];

        double3 d = make_double3(
            p_i.x - p_j.x,
            p_i.y - p_j.y,
            p_i.z - p_j.z
        );
        diff[i * numBodies + j] = d;
        diff[j * numBodies + i] = make_double3(-d.x, -d.y, -d.z);

        double distSqr_ij = d.x * d.x + d.y * d.y + d.z * d.z + 1e-10;
        distSqr[i * numBodies + j] = distSqr_ij;
        distSqr[j * numBodies + i] = distSqr_ij;

        double dist_ij = sqrt(distSqr_ij);
        dist[i * numBodies + j] = dist_ij;
        dist[j * numBodies + i] = dist_ij;
    }
}

__global__ void computeForce(double* m, double3* diff, double* distSqr, double* dist, double3* f, size_t numBodies) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double sh_m[BLOCK_SIZE];

    int i = 0;
    int sum = numBodies - 1;
    while (idx >= sum) {
        idx -= sum;
        sum -= 1;
        i += 1;
    }
    int j = i + 1 + idx;

    if (threadIdx.x < numBodies) {
        sh_m[threadIdx.x] = m[threadIdx.x];
    }
    __syncthreads();

    if (i < numBodies && j < numBodies) {
        double3 diff_ij = diff[i * numBodies + j];
        double distSqr_ij = distSqr[i * numBodies + j];

        double F = G * sh_m[i] * sh_m[j] / distSqr_ij;
        double3 force = make_double3(
            F * diff_ij.x / sqrt(distSqr_ij),
            F * diff_ij.y / sqrt(distSqr_ij),
            F * diff_ij.z / sqrt(distSqr_ij)
        );

        f[i * numBodies + j] = force;
        f[j * numBodies + i] = make_double3(-force.x, -force.y, -force.z);
    }
}

__global__ void updateBody(double3* p, double3* v, double3* f, double* m, size_t numBodies, double dt) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < numBodies) {
        double m_i = m[idx];
        double3 force = make_double3(0.0, 0.0, 0.0);
        double3 partial_force = make_double3(0.0, 0.0, 0.0);

        for (size_t j = 0; j < numBodies; ++j) {
            partial_force = f[idx * numBodies + j];
            force.x += partial_force.x;
            force.y += partial_force.y;
            force.z += partial_force.z;
        }

        double3 acceleration = make_double3(
            force.x / m_i,
            force.y / m_i,
            force.z / m_i
        );

        v[idx].x += acceleration.x * dt;
        v[idx].y += acceleration.y * dt;
        v[idx].z += acceleration.z * dt;

        p[idx].x += v[idx].x * dt;
        p[idx].y += v[idx].y * dt;
        p[idx].z += v[idx].z * dt;
    }
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
    const char* filename = "random.txt";
    const double dt = 0.1;
    const size_t numSteps = 1000;
    
    size_t numBodies = getNumBodies(filename);
    double3* h_p = new double3[numBodies];
    double3* h_v = new double3[numBodies];
    double* h_m = new double[numBodies];
    loadBodiesFromFile(filename, h_p, h_v, h_m);
    printSimlulationSummary(numBodies, dt, numSteps, G, BLOCK_SIZE, 0);

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
    checkCuda(cudaMemset(d_a, 0, numBodies * numBodies * sizeof(double3)));
    checkCuda(cudaMemset(d_diff, 0, numBodies * numBodies * sizeof(double3)));
    checkCuda(cudaMemset(d_distSqr, 0, numBodies * numBodies * sizeof(double)));

    // Record the start event
    cudaEventRecord(start);

    // Run CUDA kernel
    // int numElements = numBodies * (numBodies - 1) / 2;
    // int numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // dim3 forceBlock(FORCE_BLOCK_SIZE, FORCE_BLOCK_SIZE);
    // dim3 forceGrid((numBodies + FORCE_BLOCK_SIZE - 1) / FORCE_BLOCK_SIZE,
    //                (numBodies + FORCE_BLOCK_SIZE - 1) / FORCE_BLOCK_SIZE);
    // dim3 updateBlock(UPDATE_BLOCK_SIZE);
    // dim3 updateGrid((numBodies + updateBlock.x - 1) / updateBlock.x);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::ofstream outFile("output_data.txt", std::ios::binary);
    size_t step = 0;
    while(step < numSteps) {
        // computeDist<<<numBlocks, BLOCK_SIZE>>>(d_p, d_diff, d_distSqr, d_dist, numBodies);
        // computeForce<<<numBlocks, BLOCK_SIZE>>>(d_m, d_diff, d_distSqr, d_dist, d_f, numBodies);
        // updateBody<<<numBlocks, BLOCK_SIZE>>>(d_p, d_v, d_f, d_m, numBodies, dt);
        nBodyVelocityVerlet<<<(numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_p, d_v, d_a, d_m, d_f, numBodies, dt);
        cudaMemcpyAsync(h_p, d_p, numBodies * sizeof(double3), cudaMemcpyDeviceToHost, stream);
        for (size_t i = 0; i < numBodies; ++i) {
            outFile << h_p[i].x << " " << h_p[i].y << " " << h_p[i].z << "\n";
        }
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
    std::cout << "\n -> GPU simulation time V2: " << milliseconds << " ms\n" << std::endl;

    // Retrieve data from GPU
    checkCuda(cudaMemcpy(h_p, d_p, numBodies * sizeof(double3), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(h_v, d_v, numBodies * sizeof(double3), cudaMemcpyDeviceToHost));

    // CPU Simulation
    double3* h_pCPU = new double3[numBodies];
    double3* h_vCPU = new double3[numBodies];
    double* h_mCPU = new double[numBodies];
    double3* h_fCPU = new double3[numBodies];
    loadBodiesFromFile(filename, h_pCPU, h_vCPU, h_mCPU);

    auto startCPU = std::chrono::high_resolution_clock::now();
    step = 0;
    while(step < numSteps) {
        simulationStepCPU(h_pCPU, h_vCPU, h_mCPU, h_fCPU, numBodies, dt);
        step++;
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
