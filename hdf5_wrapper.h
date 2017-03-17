#ifndef DMT_HDF5_WRAPPER_H
#define DMT_HDF5_WRAPPER_H

#include <sstream>
#include <string>
#include <vector>

#include <hdf5.h>

template<typename T>
struct HDF5Types;

#define SPECIALIZE_HDF5_TYPE(type, hdf5_type) \
    template <> \
    struct HDF5Types<type> { \
        static hid_t map() { \
            return hdf5_type; \
        } \
    }

SPECIALIZE_HDF5_TYPE(uint8_t,  H5T_STD_U8LE);
SPECIALIZE_HDF5_TYPE(uint16_t, H5T_STD_U16LE);
SPECIALIZE_HDF5_TYPE(uint32_t, H5T_STD_U32LE);
SPECIALIZE_HDF5_TYPE(uint64_t, H5T_STD_U64LE);

struct HDF5File;

struct HDF5Dataset {
    hid_t id;
    hid_t file_id;
    hid_t data_space;
    int n_dims;
    std::vector<hsize_t> dims;
    std::string name;

    HDF5Dataset(const HDF5File& file, const std::string& name);

    template<typename T>
    void read_chunks(T* target, hsize_t* counts, hsize_t* offsets);

    template<typename T>
    void write_chunks(T* source, int n_dims, hsize_t* dims, hsize_t* counts, hsize_t* offsets);

    ~HDF5Dataset();
};

struct HDF5File {
    hid_t id;

    HDF5File(const std::string& path, unsigned flags) {
        if (flags == H5F_ACC_EXCL or flags == H5F_ACC_TRUNC) {
            this->id = H5Fcreate(path.c_str(), flags, H5P_DEFAULT, H5P_DEFAULT);
        } else if (flags == H5F_ACC_RDONLY or flags == H5F_ACC_RDWR) {
            this->id = H5Fopen(path.c_str(), flags, H5P_DEFAULT);
        } else {
            throw "Invalid access mode";
        }

        if (this->id < 0) {
            std::stringstream message;
            message << "Could not open file " << path << std::endl;
            throw message.str();
        }
    }

    HDF5Dataset operator[](const std::string& name) {
        return HDF5Dataset(*this, name);
    }

    ~HDF5File() {
        if (this->id > 0) {
            H5Fclose(this->id);
        }
    }
};

HDF5Dataset::HDF5Dataset(const HDF5File& file, const std::string& name_) : file_id(file.id), name(name_), id(-1),
                                                                           data_space(-1), n_dims(0) {
    if (!H5Lexists(this->file_id, this->name.c_str(), H5P_DEFAULT)) {
        return;
    }

    this->id = H5Dopen(this->file_id, this->name.c_str(), H5P_DEFAULT);
    if (this->id < 0) {
        std::stringstream message;
        message << "Could not open dataset " << this->name << std::endl;
        throw message.str();
    }
    this->data_space = H5Dget_space(this->id);
    this->n_dims = H5Sget_simple_extent_ndims(this->data_space);
    this->dims.resize(this->n_dims);
    H5Sget_simple_extent_dims(this->data_space, this->dims.data(), nullptr);
}

template<typename T>
void HDF5Dataset::read_chunks(T* target, hsize_t* counts, hsize_t* offsets) {
    if (this->id < 0) {
        throw "Cannot read from non-existing dataset";
    }
    hid_t memory_window = H5Screate_simple(this->n_dims, counts, nullptr);
    H5Sselect_hyperslab(this->data_space, H5S_SELECT_SET, offsets, nullptr, counts, nullptr);
    herr_t error = H5Dread(this->id, HDF5Types<T>::map(), memory_window, this->data_space, H5P_DEFAULT, target);
    if (error < 0) {
        throw "Could not read data from the dataset";
    }
}

template<typename T>
void HDF5Dataset::write_chunks(T* source, int n_dims, hsize_t* dims, hsize_t* counts, hsize_t* offsets) {
    if (this->id < 0) {
        hid_t global_memory_space = H5Screate_simple(n_dims, dims, nullptr);
        this->id = H5Dcreate(this->file_id, this->name.c_str(), HDF5Types<T>::map(), global_memory_space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        this->data_space = H5Dget_space(this->id);
        this->n_dims = n_dims;
        this->dims.assign(dims, dims + n_dims);
    }

    hid_t memory_window = H5Screate_simple(n_dims, counts, nullptr);
    H5Sselect_hyperslab(this->data_space, H5S_SELECT_SET, offsets, nullptr, counts, nullptr);
    H5Dwrite(this->id, HDF5Types<T>::map(), memory_window, this->data_space, H5P_DEFAULT, source);
}

HDF5Dataset::~HDF5Dataset() {
    if (this->id > 0) {
        H5Dclose(this->id);
    }
}

#endif //DMT_HDF5_WRAPPER_H
