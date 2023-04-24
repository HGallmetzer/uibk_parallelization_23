#include <iomanip>
#include <iostream>
#include <math.h>
#include <random>

#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

constexpr long long num_samples = 1000000000;

int main(int argc, char **argv) {

	MPI_Init(&argc, &argv);

	int comm_size = 0;
	int comm_rank = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

	if (comm_rank == 0) {
		std::cout << "Approximating PI using Monte Carlo with N=" << num_samples << " samples" << std::endl;
		std::cout << "Using MPI and " << comm_size << " ranks" << std::endl;
	}

	const long long num_samples_per_rank = num_samples / comm_size;

	long long count = 0;

	double time_start = MPI_Wtime();

	// initialize random number generator
	std::mt19937 mt(comm_rank);
	std::uniform_real_distribution<double> dist(0.0, 1.0);

	for (long long i = 0; i < num_samples_per_rank; i++) {
		const double x = dist(mt);
		const double y = dist(mt);
		if ((x * x + y * y) <= 1) {
			count++;
		}
	}

	long long total_count = 0;

	MPI_Reduce(&count, &total_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	const double pi = static_cast<double>(total_count) / num_samples * 4.0;

	double time_end = MPI_Wtime();

	if (comm_rank == 0) {
		std::cout << std::setprecision(16) << "PI is approximately " << pi << std::endl;
		std::cout << "Computation took " << time_end - time_start << " seconds" << std::endl;
	}

	MPI_Finalize();

	return EXIT_SUCCESS;
}
