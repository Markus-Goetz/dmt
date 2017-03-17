#ifndef DMT_HDF5_WRAPPER_H
#define DMT_HDF5_WRAPPER_H

#include <sstream>
#include <string>
#include <vector>

#include <hdf5.h>

template <typename T>
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

struct HDF5File;

struct HDF5Dataset {
    hid_t id;
    hid_t data_space;
    int n_dims;
    std::vector<hsize_t> dims;

    HDF5Dataset(const HDF5File& file, const std::string& name);
    template <typename T>
    void read_chunks(T* target, hsize_t* counts, hsize_t* offsets);
    ~HDF5Dataset();
};

struct HDF5File {
    hid_t id;

    HDF5File(const std::string& path, unsigned int flags) {
        this->id = H5Fopen(path.c_str(), flags, H5P_DEFAULT);
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

HDF5Dataset::HDF5Dataset(const HDF5File& file, const std::string& name) {
    this->id = H5Dopen1(file.id, name.c_str());
    if (this->id < 0) {
        std::stringstream message;
        message << "Could not open dataset " << name << std::endl;
        throw message.str();
    }
    this->data_space = H5Dget_space(this->id);
    this->n_dims = H5Sget_simple_extent_ndims(this->data_space);
    this->dims.resize(this->n_dims);
    H5Sget_simple_extent_dims(this->data_space, this->dims.data(), nullptr);
}

template <typename T>
void HDF5Dataset::read_chunks(T* target, hsize_t* counts, hsize_t* offsets) {
    hid_t type = H5Dget_type(this->id);
    hid_t memory_window = H5Screate_simple(this->n_dims, counts, nullptr);
    H5Sselect_hyperslab(this->data_space, H5S_SELECT_SET, offsets, nullptr, counts, nullptr);
    herr_t error = H5Dread(this->id, HDF5Types<T>::map(), memory_window, this->data_space, H5P_DEFAULT, target);
    if (error < 0) {
        throw "Could not read data from the dataset";
    }
}

HDF5Dataset::~HDF5Dataset() {
    if (this->id > 0) {
        H5Dclose(this->id);
    }
}

#endif //DMT_HDF5_WRAPPER_H
