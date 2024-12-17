#ifndef NBODY_CUH
#define NBODY_CUH

#include <iomanip>
#include <fstream>
#include <iostream>

const double G = 6.67430e-11;

void loadBodiesFromFile(const char* filename, double3* p, double3* v, double* m) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Unable to open: " << filename << std::endl;
        return;
    }

    double mass, x, y, z, vx, vy, vz;
    size_t i = 0;
    while (file >> mass >> x >> y >> z >> vx >> vy >> vz) {
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

size_t getNumBodies(const char* filename) {
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

void printSimlulationSummary(const size_t numBodies, const double dt, const size_t numSteps, const double G, const size_t blockSize, const size_t gridSize) {
    std::cout << "------------------------------------------" << std::endl
                << "| " << std::setw(30) << "Simulation Summary" << std::setw(10) << " |" << std::endl
                << "------------------------------------------" << std::endl;
    std::cout << std::left
                << std::setw(30) << "| Number of Bodies" << std::setw(10) << numBodies << " |" << std::endl
                << std::setw(30) << "| Time Step (dt)" << std::setw(10) << dt << " |" << std::endl
                << std::setw(30) << "| Number of Steps" << std::setw(10) << numSteps << " |" << std::endl
                << std::setw(30) << "| Gravitational Constant (G)" << std::setw(10) << G << " |" << std::endl
                << std::setw(30) << "| Block Size" << std::setw(10) << blockSize << " |" << std::endl
                << std::setw(30) << "| Grid Size" << std::setw(10) << gridSize << " |" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
}
#endif // NBODY_CUH
