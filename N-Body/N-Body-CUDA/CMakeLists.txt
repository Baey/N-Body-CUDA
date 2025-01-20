cmake_minimum_required(VERSION 3.10)

project(NBodySimulation CUDA)

set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

set(SOURCES_SIM_APP
    src/nbody_cuda_sim.cu
    src/kernel.cu
    src/nbody_utils.cu
)

set(SOURCES_BENCHMARK_APP
    src/nbody_cuda_benchmark.cu
    src/kernel.cu
    src/nbody_utils.cu
)

set(HEADERS
    include/cuda_utils.cuh
    include/nbody_utils.cuh
    include/kernel.cuh
)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(nbody_sim_cuda ${SOURCES_SIM_APP} ${HEADERS})
add_executable(nbody_cuda_benchmark ${SOURCES_BENCHMARK_APP} ${HEADERS})