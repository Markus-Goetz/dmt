#include <cstdint>
#include <iostream>

#include <mpi.h>

#include <image.h>
#include <distributed_max_tree.h>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Image<uint8_t> image;
    try {
        image.read("../examples.h5", "small", MPI_COMM_WORLD);
    } catch (const std::string& message) {
        if (rank == 0)
            std::cerr << message << std::endl;
        exit(1);
    }

    DistributedMaxTree dmt;
    Parents parents = dmt.compute(image);
    std::cout << parents << std::endl;
    
    return MPI_Finalize();
}
