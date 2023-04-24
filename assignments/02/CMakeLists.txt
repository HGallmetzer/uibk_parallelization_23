cmake_minimum_required(VERSION 3.9)

project(assignments C CXX)

find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -fsanitize=undefined -Wall -Wextra -pedantic")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(stencil_mpi stencil_mpi.cpp)
target_link_libraries(stencil_mpi PUBLIC MPI::MPI_C)

add_executable(pi_mpi pi_mpi.cpp)
target_link_libraries(pi_mpi PUBLIC MPI::MPI_C)
