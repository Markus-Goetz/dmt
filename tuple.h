#ifndef TUPLE_H
#define TUPLE_H

#include <iostream>
#include <limits>
#include <sstream>
#include <tuple>
#include <vector>

template<typename T, typename U=Parents::type>
struct Tuple {
    static constexpr size_t bytes = 2 * sizeof(T) + 2 * sizeof(U);

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

    uint64_t htons(uint64_t value) const {
        value = ((value << 8) & 0xFF00FF00FF00FF00ULL ) | ((value >> 8) & 0x00FF00FF00FF00FFULL );
        value = ((value << 16) & 0xFFFF0000FFFF0000ULL ) | ((value >> 16) & 0x0000FFFF0000FFFFULL );
        return (value << 32) | ((value >> 32) & 0xFFFFFFFFULL);
    }

    void pack(uint8_t* destination) const {
        destination[0] = this->color;
        *reinterpret_cast<uint64_t*>(destination + 1) = htons(this->from);
        destination[9] = this->neighbor_color;
        *reinterpret_cast<uint64_t*>(destination + 10) = htons(this->to);
    }
};

template<typename T, typename U=Parents::type>
using Tuples = std::vector<Tuple<T, U> >;

template<typename T, typename U=Parents::type>
using TupleBuckets = std::vector<Tuples<T, U> >;

#endif // TUPLE_H
