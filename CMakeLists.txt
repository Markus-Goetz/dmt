cmake_minimum_required(VERSION 3.3)
project(dmt)

find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED)

    #set(CMAKE_CXX_COMPILER "scalasca")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -instrument ${MPI_CXX_COMPILER}")

set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_CXX_STANDARD 11)

include_directories(.)
include_directories(${HDF5_INCLUDE_DIRS})
set(SOURCE_FILES
    distributed_max_tree.h
    endpoint.h
    image.h
    main.cpp
    tuple.h
    mpi_wrapper.h
    hdf5_wrapper.h
    util.h
)

add_executable(dmt ${SOURCE_FILES})

link_directories(${HDF5_LIBRARY_DIRS})
target_link_libraries(dmt ${HDF5_LIBRARIES})
