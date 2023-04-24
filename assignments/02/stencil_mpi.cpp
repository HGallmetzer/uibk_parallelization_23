#include <iostream>

#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

#include <string>
#include <vector>

constexpr std::size_t size_domain = 65536;
constexpr std::size_t size_halo = 1;
constexpr std::size_t timesteps = 5000;
constexpr std::size_t output_resolution = 120;

using Datatype = double;
using Domain = std::vector<Datatype>;
using SubDomain = Domain;

void printTemperature(const Domain &domain, const std::size_t size_subdomain);

int verifyTemperature(const Domain &domain);

inline constexpr std::size_t adjustIndexForHalo(std::size_t index) { return index + size_halo; }

inline constexpr std::size_t adjustSizeForHalo(std::size_t size) { return size - size_halo; }

int main(int argc, char **argv) {

	MPI_Init(&argc, &argv);

	int comm_size = 0;
	int comm_rank = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

	if (comm_rank == 0) {
		std::cout << "Computing heat-distribution for room size N=" << size_domain << " for T=" << timesteps << std::endl;
		std::cout << "Using MPI and " << comm_size << " ranks" << std::endl;
	}

	if ((size_domain % comm_size) != 0) {
		if (comm_rank == 0) {
			std::cout << "Please ensure that the size is evenly divisible by the number of ranks. "
			             "Exiting."
			          << std::endl;
		}
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	double time_start = MPI_Wtime();

	const std::size_t size_subdomain = size_domain / comm_size;

	const int left_neighbor = (comm_rank - 1 + comm_size) % comm_size;
	const int right_neighbor = (comm_rank + 1) % comm_size;

	Domain domain;
	const std::size_t source_x = size_domain / 2;

	if (comm_rank == 0) {
		domain = Domain(size_domain);

		for (auto &cell : domain) {
			cell = 273.0;
		}

		std::cout << "Heat Source is at " << source_x << std::endl;
		domain[source_x] = 273 + 60;

		std::cout << "Initial:\t";
		printTemperature(domain, size_subdomain);
		std::cout << std::endl;
	}

	SubDomain subdomain_a = SubDomain(size_subdomain + size_halo * 2, 273.0);
	SubDomain subdomain_b = SubDomain(size_subdomain + size_halo * 2, 273.0);

	MPI_Scatter(domain.data(), size_subdomain, MPI_DOUBLE, &subdomain_a[1], size_subdomain, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	for (std::size_t t = 0; t < timesteps; t++) {

		// send to right and receive from left
		MPI_Sendrecv(&subdomain_a[subdomain_a.size() - 2], 1, MPI_DOUBLE, right_neighbor, 42, &subdomain_a[0], 1, MPI_DOUBLE, left_neighbor, 42, MPI_COMM_WORLD,
		             MPI_STATUS_IGNORE);
		// send to left and receive from right
		MPI_Sendrecv(&subdomain_a[1], 1, MPI_DOUBLE, left_neighbor, 42, &subdomain_a[subdomain_a.size() - 1], 1, MPI_DOUBLE, right_neighbor, 42, MPI_COMM_WORLD,
		             MPI_STATUS_IGNORE);

		for (std::size_t x = 1; x <= size_subdomain; x++) {
			if ((source_x / size_subdomain) == static_cast<std::size_t>(comm_rank) && x == adjustIndexForHalo(source_x % size_subdomain)) {
				subdomain_b[x] = subdomain_a[x];
				continue;
			}

			Datatype value_left = subdomain_a[x - 1];
			Datatype value_center = subdomain_a[x];
			Datatype value_right = subdomain_a[x + 1];

			subdomain_b[x] = value_center + 0.2 * (value_left + value_right + (-2.0 * value_center));
		}

		std::swap(subdomain_a, subdomain_b);

		if ((t % 1000) == 0) {
			MPI_Gather(&subdomain_a[1], size_subdomain, MPI_DOUBLE, domain.data(), size_subdomain, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			if (comm_rank == 0) {
				std::cout << "Step t=" << t << "\t";
				printTemperature(domain, size_subdomain);
				std::cout << std::endl;
			}
		}
	}

	MPI_Gather(&subdomain_a[1], size_subdomain, MPI_DOUBLE, domain.data(), size_subdomain, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Finalize();

	double time_end = MPI_Wtime();

	int verification_result = EXIT_SUCCESS;
	if (comm_rank == 0) {
		verification_result = verifyTemperature(domain);
		std::cout << "Computation took " << time_end - time_start << " seconds" << std::endl;
	}

	return verification_result;
}

void printTemperature(const Domain &domain, const std::size_t size_subdomain) {
	const std::string colors = " .-:=+*^X#%@";

	constexpr Datatype max = 273 + 30;
	constexpr Datatype min = 273 + 0;

	// step size in each dimension
	const std::size_t step_size = domain.size() / output_resolution;

	// left border
	std::cout << "X";

	for (std::size_t i = 0; i < output_resolution; i++) {

		// rank boundary
		if (i != 0 && (i % (output_resolution / (domain.size() / size_subdomain))) == 0) {
			std::cout << "|";
		}

		// get max temperature in this tile
		Datatype cur_max = 0;
		for (std::size_t x = step_size * i; x < step_size * i + step_size; x++) {
			cur_max = (cur_max < domain[x]) ? domain[x] : cur_max;
		}
		Datatype temp = cur_max;

		// pick the 'color'
		int c = ((temp - min) / (max - min)) * colors.length();
		c = (c >= static_cast<int>(colors.length())) ? colors.length() - 1 : ((c < 0) ? 0 : c);

		// print the average temperature
		std::cout << colors[c];
	}

	// right border
	std::cout << "X";
}

int verifyTemperature(const Domain &domain) {
	for (std::size_t x = 0; x < domain.size(); x++) {
		if (domain[x] < 273.0 || domain[x] > 273.0 + 60) {
			std::cout << "Verification failed, grid[" << x << "]=" << domain[x] << std::endl;
			return EXIT_FAILURE;
		}
	}
	std::cout << "Verification succeeded" << std::endl;
	return EXIT_SUCCESS;
}