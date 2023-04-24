# Assignment 2

The goal of this assignment is to get you acquainted with working on a distributed memory cluster as well as obtaining, illustrating, and interpreting measurement data.

### Description

We provided two simple parallel programs that you often find (in much more complex fashion) in real-world simulations, a structured grid application (`stencil_mpi.cpp`) and a Monte Carlo approximation (`pi_mpi.cpp`). For this assignment, you should compile these codes and run them with various degrees of parallelism (numbers of processes, or *ranks* in MPI-speak, or *tasks* in SLURM-speak).

### Tasks

- Compile the source codes and make sure that they execute properly. Which compiler flags did you use and why?
- Prepare a SLURM script that runs the programs for 1, 2, 4, 8, 16, 32 and 64 ranks.
- Create a table and figures that illustrate the measured data and study them. What effects can you observe?
- How stable are the measurements when running the experiments multiple times?
- Insert the measured time for 1 and 64 ranks, for `N=65536` and `T=5000` for the stencil, and `N=1e9` for the Monte Carlo approximation into the provided comparison spreadsheet.

### Hints

- You are free to choose how you compile the codes, but e.g. you can use `cmake` using the provided `CMakeLists.txt`:
    - load relatively modern versions of cmake, gcc and OpenMPI: `module load cmake/3.24.3-gcc-12.2.0-lljyybc gcc/10.3.0 openmpi/4.0.3`
    - compile the codes: `mkdir build && cmake .. -DCMAKE_BUILD_TYPE=Release`
    - done
