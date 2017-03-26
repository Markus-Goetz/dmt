#ifndef DMT_MPI_WRAPPER_H
#define DMT_MPI_WRAPPER_H

#include <cstdint>

#include <mpi.h>

template <typename T>
struct MPI_Types;

#define SPECIALIZE_MPI_TYPE(type, mpi_type) \
    template <> \
    struct MPI_Types<type> { \
        static constexpr MPI_Datatype map() {\
            return mpi_type; \
        } \
    }
    
SPECIALIZE_MPI_TYPE(uint8_t,  MPI_BYTE);
SPECIALIZE_MPI_TYPE(uint16_t, MPI_UNSIGNED_SHORT);
SPECIALIZE_MPI_TYPE(uint32_t, MPI_UNSIGNED);
SPECIALIZE_MPI_TYPE(uint64_t, MPI_UNSIGNED_LONG);

#endif // DMT_MPI_WRAPPER_H
