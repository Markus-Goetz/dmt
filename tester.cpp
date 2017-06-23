#include <cstdint>
#include <iostream>
#include <map>

#include <mpi.h>

#include "distributed_max_tree.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int i = 1; i <= 9; i++) {
        Image<uint8_t> image;
        Image<uint64_t> parent;
        std::vector<hsize_t> total_dimensions;

        try {
            total_dimensions = image.read("../img/input_" + std::to_string(i) + ".h5", "test", MPI_COMM_WORLD);
        } catch (const std::string& message) {
            if (rank == 0) {
                std::cerr << message << std::endl;
            }
            return MPI_Finalize();
        }

        try {
            total_dimensions = parent.read("../img/parent_" + std::to_string(i) + ".h5", "test", MPI_COMM_WORLD);
        } catch (const std::string& message) {
            if (rank == 0) {
                std::cerr << message << std::endl;
            }
            return MPI_Finalize();
        }

        DistributedMaxTree dmt;
        Parents parents = dmt.compute(image);

        bool check = false;
        std::map<uint64_t, uint64_t> mismatches;
        size_t offset = (rank + 1 < size) ? parents.width() : 0;
        size_t end = parents.size()  - offset;

        for (size_t j = 0; j < end; ++j) {
            if (parents[j] != parent[j]) {
                mismatches[parents[j]] = parent[j];
                check = true;
            }
        }

        for (auto match : mismatches) {
            std::cout << "K: " << match.first << " V: " << match.second << std::endl;
        }

        MPI_Allreduce(MPI_IN_PLACE, &check, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        if (check and rank == 0) {
            std::cout << "Parent " << i << " does not match! " << std::endl;
        } else if (rank == 0) {
            std::cout << "Parent " << i << " [ok]" << std::endl;
        }
    }

    return MPI_Finalize();
}
