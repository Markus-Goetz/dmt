#include <cstdint>
#include <iostream>

#include <mpi.h>

#include "distributed_max_tree.h"
#include "image.h"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 5) {
        if (rank == 0) {
            std::cout << "Usage: " << argv[0] << " <IN_FILE> <IN_DATASET> <OUT_FILE> <OUT_DATASET>" << std::endl;
        }
        return MPI_Finalize();
    }

    Image<uint8_t> image;
    std::vector<hsize_t> total_dimensions;
    try {
        total_dimensions = image.read(argv[1], argv[2], MPI_COMM_WORLD);
    } catch (const std::string& message) {
        if (rank == 0) {
            std::cerr << message << std::endl;
        }
        return MPI_Finalize();
    }

    DistributedMaxTree dmt;
    Parents parents = dmt.compute(image);

    try {
        parents.write(argv[3], argv[4], MPI_COMM_WORLD, total_dimensions);
    } catch (const std::string& message) {
        if (rank == 0) {
            std::cerr << message << std::endl;
        }
        return MPI_Finalize();
    }
    
    return MPI_Finalize();
}
