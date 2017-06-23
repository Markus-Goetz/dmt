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

    static void free_mpi_type(MPI_Datatype* type) {
        MPI_Type_free(type);
    }
};

template <typename U=Parents::type>
using AreaEndpoints = std::array<AreaEndpoint<U>, 2>;

template<typename T, typename U=Parents::type>
using Endpoints = std::array<Tuple<T, U>, 2>;

#endif // ENDPOINT_H
