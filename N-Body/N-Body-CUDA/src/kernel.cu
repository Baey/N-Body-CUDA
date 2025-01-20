#include <cuda_runtime.h>
#include "kernel.cuh"
#include "nbody_utils.cuh"


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

__global__ void nBody(double3* p, double3* v, double* m, double3* f, size_t numBodies, double dt) {
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

__global__ void computeDist(double3* p, double3* diff, double* distSqr, double* dist, size_t numBodies) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ double3 sh_p[];

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

    extern __shared__ double sh_m[];

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