#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>

__global__ void nBodyVelocityVerlet(double3* p, double3* v, double3* a, double* m, double3* f, size_t numBodies, double dt);

__global__ void nBody(double3* p, double3* v, double* m, double3* f, size_t numBodies, double dt);

__global__ void computeDist(double3* p, double3* diff, double* distSqr, double* dist, size_t numBodies);

__global__ void computeForce(double* m, double3* diff, double* distSqr, double* dist, double3* f, size_t numBodies);

__global__ void updateBody(double3* p, double3* v, double3* f, double* m, size_t numBodies, double dt);

#endif