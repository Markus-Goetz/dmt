#include <cstdint>
#include <iostream>

#include <mpi.h>

#include <image.h>
#include <distributed_max_tree.h>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    size_t width = 12;
    size_t height = 8;
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

//    if (rank == 0)
//    {
//        Image<uint8_t> image({
//                1, 4, 2, 1,
//        }, 1, 4);
//
//        DistributedMaxTree dmt;
//        dmt.compute(image);
//    }
//    else
//    {
//        Image<uint8_t> image({
//             1, 2, 1, 2
//        }, 1, 4);

    if (rank == 0) {
        Image<uint8_t> image({
            1, 1, 0, 0, 0, 0, 0, 0,
            1, 1, 0, 0, 0, 0, 0, 0,
            3, 3, 0, 4, 4, 3, 3, 0,
            3, 3, 0, 0, 0, 0, 0, 0,
            4, 4, 0, 4, 0, 0, 0, 0,
            3, 3, 4, 4, 0, 0, 0, 0,
            4, 4, 0, 0, 0, 0, 0, 0,
        }, 8, 7);

        DistributedMaxTree dmt;
        dmt.compute(image);
    } else {
        Image<uint8_t> image({
            4, 4, 0, 0, 0, 0, 0, 0,
            4, 4, 0, 0, 0, 0, 0, 0,
            2, 2, 0, 0, 0, 0, 0, 0,
            2, 2, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 1, 0,
            2, 2, 1, 1, 2, 2, 1, 2
        }, 8, 7);

//    if (rank == 0)
//    {
//        Image<uint8_t> image({
//            0,0,0,0,0,0,0,0,0,0,0,0,
//            0,0,0,4,4,4,4,0,0,0,0,0,
//            0,0,0,4,4,4,4,0,0,0,0,0,
//            0,0,0,4,4,5,5,5,5,0,0,0,
//            0,0,4,4,4,5,5,5,5,0,0,0,
//            1,1,1,1,1,5,5,4,4,4,4,1,
//            1,1,1,1,1,5,5,4,4,4,4,1,
//            1,1,1,1,1,1,1,4,4,4,4,6
//        }, width, height);
//
//        DistributedMaxTree dmt;
//        dmt.compute(image);
//    }
//    else
//    {
//        Image<uint8_t> image({
//            1,1,1,1,1,1,1,4,4,4,4,6,
//            1,1,1,1,1,1,1,4,4,4,4,6,
//            1,1,1,1,1,1,1,4,4,4,4,6,
//            1,1,1,1,1,1,1,1,6,6,6,6,
//            1,1,1,1,1,1,1,1,6,6,6,6,
//            1,1,1,1,1,1,1,1,6,6,6,6,
//            0,0,0,0,4,4,4,4,1,1,1,4,
//            0,0,0,0,4,4,4,4,1,1,1,1
//        }, width, height);

        DistributedMaxTree dmt;
        dmt.compute(image);
    }
    
    return MPI_Finalize();
}
