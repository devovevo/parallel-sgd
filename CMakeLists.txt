cmake_minimum_required(VERSION 3.14)
project(parallel-sgd)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)

# netID
# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

add_executable(serial serial_mlp.cpp common.cpp common.hpp activations.h data.cpp data.hpp)
target_link_libraries(serial PRIVATE ${BLAS_LIBRARIES})

add_executable(all_reduce sync_param_mlp.cpp common.cpp common.hpp activations.h data.cpp data.hpp)
target_link_libraries(all_reduce PRIVATE MPI::MPI_CXX ${BLAS_LIBRARIES})
