CMAKE_MINIMUM_REQUIRED(VERSION 3.10 FATAL_ERROR)
PROJECT(dmt)

# set c++ properties
ENABLE_LANGUAGE(CXX)
set(CMAKE_BUILD_TYPE RELEASE)
SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS "-Ofast -Wall -Wextra -pedantic")

# add the include and source directories
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

# detect mpi
FIND_PACKAGE(MPI REQUIRED)
SET(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
INCLUDE_DIRECTORIES(${MPI_CXX_INCLUDE_PATH})

# detect hdf5
FIND_PACKAGE(HDF5 REQUIRED)
INCLUDE_DIRECTORIES("${HDF5_INCLUDE_DIRS}")
LINK_DIRECTORIES(${HDF5_LIBRARY_DIRS})

# detect C++14 threads
FIND_PACKAGE(Threads REQUIRED)

# define the source files
SET(SOURCE_FILES
    distributed_max_tree.h
    endpoints.h
    image.h
    main.cpp
    tuple.h
    mpi_wrapper.h
    hdf5_wrapper.h
    util.h
)

# main executable definition
ADD_EXECUTABLE(dmt src/main.cpp)
TARGET_LINK_LIBRARIES(dmt ${HDF5_LIBRARIES})
TARGET_LINK_LIBRARIES(dmt Threads::Threads)

# test suite executable
ADD_EXECUTABLE(tester src/tester.cpp)
TARGET_LINK_LIBRARIES(tester ${HDF5_LIBRARIES})
TARGET_LINK_LIBRARIES(tester Threads::Threads)
