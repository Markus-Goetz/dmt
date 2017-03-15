cmake_minimum_required(VERSION 3.3)
project(dmt)

find_package(MPI REQUIRED)

set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
set(CMAKE_CXX_STANDARD 11)

include_directories(.)
set(SOURCE_FILES
        distributed_max_tree.h
        endpoint.h
        image.h
        main.cpp
        tuple.h
        util.h)

add_executable(dmt ${SOURCE_FILES})
