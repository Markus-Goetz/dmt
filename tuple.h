#ifndef TUPLE_H
#define TUPLE_H

#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <tuple>
#include <vector>

#include "mpi_wrapper.h"

template<typename T, typename U=Parents::type>
struct Tuple {
    T color = std::numeric_limits<T>::max();
    U from = std::numeric_limits<U>::max();
    T neighbor_color = std::numeric_limits<T>::max();
    U to = std::numeric_limits<U>::max();

    Tuple() = default;

    Tuple(T color_, U from_, T neighbor_color_, U to_)
            : color(color_), from(from_), neighbor_color(neighbor_color_), to(to_) {}

    friend bool operator<(const Tuple& lhs, const Tuple& rhs) {
        return std::tie(lhs.color, lhs.from, lhs.neighbor_color, lhs.to) <
               std::tie(rhs.color, rhs.from, rhs.neighbor_color, rhs.to);
    }

    friend inline bool operator>(const Tuple& lhs, const Tuple& rhs) {
        return rhs < lhs;
    }

    friend inline bool operator<=(const Tuple& lhs, const Tuple& rhs) {
        return !(lhs > rhs);
    }

    friend inline bool operator>=(const Tuple& lhs, const Tuple& rhs) {
        return !(lhs < rhs);
    }

    friend inline bool operator==(const Tuple& lhs, const Tuple& rhs) {
        return std::tie(lhs.color, lhs.from, lhs.neighbor_color, lhs.to) ==
               std::tie(rhs.color, rhs.from, rhs.neighbor_color, rhs.to);
    }

    friend std::ostream& operator<<(std::ostream& os, const Tuple& tuple) {
        std::stringstream ss;
        ss << "<" << +tuple.color
           << ", " << +tuple.from
           << ", " << +tuple.neighbor_color
           << ", " << +tuple.to
           << ">";
        return os << ss.str();
    }

    static void create_mpi_type(MPI_Datatype* type) {
        Tuple <T, U> tuple;

        const int parts = 4;
        int counts[parts] = {1, 1, 1, 1};
        MPI_Aint displacements[parts] = {};

        MPI_Aint base, offset;
        MPI_Get_address(&tuple, &base);
        MPI_Get_address(&tuple.color, &offset);
        displacements[0] = offset - base;
        MPI_Get_address(&tuple.from, &offset);
        displacements[1] = offset - base;
        MPI_Get_address(&tuple.neighbor_color, &offset);
        displacements[2] = offset - base;
        MPI_Get_address(&tuple.to, &offset);
        displacements[3] = offset - base;

        MPI_Datatype types[parts] = {
                MPI_Types<T>::map(),
                MPI_Types<U>::map(),
                MPI_Types<T>::map(),
                MPI_Types<U>::map()
        };

        MPI_Type_create_struct(parts, counts, displacements, types, type);
        MPI_Type_commit(type);
    }

    static void free_mpi_type(MPI_Datatype* type) {
        MPI_Type_free(type);
    }
};

template<typename T, typename U=Parents::type>
using Tuples = std::vector<Tuple<T, U> >;

template<typename T, typename U=Parents::type>
using TupleBuckets = std::map<T, Tuples<T, U> >;

#endif // TUPLE_H