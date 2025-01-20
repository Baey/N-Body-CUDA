#include <iomanip>
#include <fstream>
#include <iostream>
#include "nbody_utils.cuh"

#include <cuda_runtime.h>

void loadBodiesFromFile(const std::string& filename, double3* p, double3* v, double* m) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Unable to open: " << filename << std::endl;
        return;
    }

    double mass, r, x, y, z, vx, vy, vz;
    size_t i = 0;
    while (file >> mass >> r >> x >> y >> z >> vx >> vy >> vz) {
        if (mass <= 0) {
            std::cerr << "Invalid mass value: " << mass << std::endl;
            continue;
        }
        p[i] = make_double3(x, y, z);
        v[i] = make_double3(vx, vy, vz);
        m[i] = mass;
        i++;
    }
    file.close();
}

size_t getNumBodies(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Unable to open: " << filename << std::endl;
        return 0;
    }

    size_t numBodies = 0;
    while (!file.eof()) {
        std::string line;
        std::getline(file, line);
        numBodies++;
    }
    file.close();
    return numBodies;
}

void printSimulationSummary(const size_t numBodies, const double dt, const size_t numSteps, const double G_const, const size_t blockSize, const size_t gridSize) {
    std::cout << "------------------------------------------" << std::endl
                << "| " << std::setw(30) << "Simulation Summary" << std::setw(10) << " |" << std::endl
                << "------------------------------------------" << std::endl;
    std::cout << std::left
                << std::setw(30) << "| Number of Bodies" << std::setw(10) << numBodies << " |" << std::endl
                << std::setw(30) << "| Time Step (dt)" << std::setw(10) << dt << " |" << std::endl
                << std::setw(30) << "| Number of Steps" << std::setw(10) << numSteps << " |" << std::endl
                << std::setw(30) << "| Gravitational Constant (G)" << std::setw(10) << G_const << " |" << std::endl
                << std::setw(30) << "| Block Size" << std::setw(10) << blockSize << " |" << std::endl
                << std::setw(30) << "| Grid Size" << std::setw(10) << gridSize << " |" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
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

void nbodyStepCPU(double3* p, double3* v, double* m, double3* f, size_t numBodies, double dt) {
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

void nbodyStepVelocityVerletCPU(double3* p, double3* v, double3* a, double* m, double3* f, size_t numBodies, double dt) {
    for (size_t i = 0; i < numBodies; i++) {
        double3 p_new = make_double3(
            p[i].x + v[i].x * dt + 0.5 * a[i].x * dt * dt,
            p[i].y + v[i].y * dt + 0.5 * a[i].y * dt * dt,
            p[i].z + v[i].z * dt + 0.5 * a[i].z * dt * dt
        );
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
        double3 a_new = make_double3(force.x / m[i], force.y / m[i], force.z / m[i]);
        f[i] = force;
        double3 v_new = make_double3(
            v[i].x + 0.5 * (a[i].x + a_new.x) * dt,
            v[i].y + 0.5 * (a[i].y + a_new.y) * dt,
            v[i].z + 0.5 * (a[i].z + a_new.z) * dt
        );

        p[i] = p_new;
    }
}
