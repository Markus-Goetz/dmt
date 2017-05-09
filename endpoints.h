#ifndef ENDPOINT_H
#define ENDPOINT_H

#include <array>
#include <iostream>
#include <limits>
#include <sstream>
#include <tuple>

#include <mpi.h>

#include "image.h"
#include "mpi_wrapper.h"
#include "tuple.h"

template <typename U=Parents::type>
struct AreaEndpoint {
    U from;
    U to;
    U canonical_to;

    AreaEndpoint() :
            from(std::numeric_limits<U>::max()),
            to(std::numeric_limits<U>::max()),
            canonical_to(std::numeric_limits<U>::max())
    {}

    AreaEndpoint(U from_, U to_, U current_to_) : from(from_), to(to_), canonical_to(current_to_) {};

    friend std::ostream& operator<<(std::ostream& os, const AreaEndpoint& endpoint) {
        std::stringstream ss;
        ss << "<" <<  +endpoint.from
           << ", " << +endpoint.to
           << ", " << +endpoint.canonical_to
           << ">";
        return os << ss.str();
    }

    static void create_mpi_type(MPI_Datatype* type) {
        AreaEndpoint<U> endpoint;

        const int parts = 3;
        int counts[parts] = {1, 1, 1};
        MPI_Aint displacements[parts] = {};

        MPI_Aint base, offset;
        MPI_Get_address(&endpoint, &base);
        MPI_Get_address(&endpoint.from, &offset);
        displacements[0] = offset - base;
        MPI_Get_address(&endpoint.to, &offset);
        displacements[1] = offset - base;
        MPI_Get_address(&endpoint.canonical_to, &offset);
        displacements[2] = offset - base;

        MPI_Datatype types[parts] = {
                MPI_Types<U>::map(),
                MPI_Types<U>::map(),
                MPI_Types<U>::map()
        };

        MPI_Type_create_struct(parts, counts, displacements, types, type);
        MPI_Type_commit(type);
    }
};

template<typename T, typename U=Parents::type>
struct Endpoint {
    T color = std::numeric_limits<T>::max();
    U from = std::numeric_limits<U>::max();
    T start_root_color = std::numeric_limits<T>::max();
    U start_root_to = std::numeric_limits<U>::max();
    U current_from = std::numeric_limits<U>::max();
    T current_root_color = std::numeric_limits<T>::max();
    U current_root_to = std::numeric_limits<U>::max();

    Endpoint() = default;

    friend std::ostream& operator<<(std::ostream& os, const Endpoint& endpoint) {
        std::stringstream ss;
        ss << "<" << +endpoint.color
           << ", " << +endpoint.from
           << ", " << +endpoint.start_root_color
           << ", " << +endpoint.start_root_to
           << ", " << +endpoint.current_root_color
           << ", " << +endpoint.current_root_to
           << ", " << +endpoint.current_from
           << ">";
        return os << ss.str();
    }

    static void create_mpi_type(MPI_Datatype* type) {
        Endpoint<T, U> endpoint;

        const int parts = 7;
        int counts[parts] = {1, 1, 1, 1, 1, 1, 1};
        MPI_Aint displacements[parts] = {};

        MPI_Aint base, offset;
        MPI_Get_address(&endpoint, &base);
        MPI_Get_address(&endpoint.color, &offset);
        displacements[0] = offset - base;
        MPI_Get_address(&endpoint.from, &offset);
        displacements[1] = offset - base;
        MPI_Get_address(&endpoint.start_root_color, &offset);
        displacements[2] = offset - base;
        MPI_Get_address(&endpoint.start_root_to, &offset);
        displacements[3] = offset - base;
        MPI_Get_address(&endpoint.current_from, &offset);
        displacements[4] = offset - base;
        MPI_Get_address(&endpoint.current_root_color, &offset);
        displacements[5] = offset - base;
        MPI_Get_address(&endpoint.current_root_to, &offset);
        displacements[6] = offset - base;

        MPI_Datatype types[parts] = {
                MPI_Types<T>::map(),
                MPI_Types<U>::map(),
                MPI_Types<T>::map(),
                MPI_Types<U>::map(),
                MPI_Types<U>::map(),
                MPI_Types<T>::map(),
                MPI_Types<U>::map()
        };

        MPI_Type_create_struct(parts, counts, displacements, types, type);
        MPI_Type_commit(type);
    }
};

template <typename U=Parents::type>
using AreaEndpoints = std::array<AreaEndpoint<U>, 2>;

template<typename T, typename U=Parents::type>
using Endpoints = std::array<Endpoint<T, U>, 2>;

#endif // ENDPOINT_H
