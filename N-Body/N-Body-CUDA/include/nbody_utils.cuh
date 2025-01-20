#ifndef NBODY_UTILS_H
#define NBODY_UTILS_H

#include <iostream>
#include <cuda_runtime.h>

#define G 6.67430e-11f

void loadBodiesFromFile(const std::string& filename, double3* h_p, double3* h_v, double* h_m);

size_t getNumBodies(const std::string& filename);

void printSimulationSummary(const size_t numBodies, const double dt, const size_t numSteps, const double G_const, const size_t blockSize, const size_t gridSize);

void printMatrix(const char* name, double* d_matrix, size_t rows, size_t cols);

void printMatrix3(const char* name, double3* d_matrix, size_t rows, size_t cols);

void nbodyStepCPU(double3* p, double3* v, double* m, double3* f, size_t numBodies, double dt);

void nbodyStepVelocityVerletCPU(double3* p, double3* v, double3* a, double* m, double3* f, size_t numBodies, double dt);

#endif