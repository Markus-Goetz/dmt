#ifndef IMAGE_H
#define IMAGE_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "hdf5_wrapper.h"

template<typename T>
class Image {
protected:
    std::vector<T> pixels_;

public:
    static const T infinity;

    size_t width_;
    size_t height_;
    typedef T type;

    Image() : width_(0), height_(0) {}

    Image(size_t width, size_t height) : pixels_(width * height), width_(width), height_(height) {}

    Image(T fill, size_t width, size_t height) : pixels_(width * height, fill), width_(width), height_(height) {}

    Image(std::initializer_list<T> pixels, size_t width, size_t height)
            : pixels_(pixels), width_(width), height_(height) {
        assert(this->pixels_.size() == this->width_ * this->height_);
    }

    inline T& operator[](size_t index) {
        return this->pixels_[index];
    }

    inline const T& operator[](size_t index) const {
        return this->pixels_[index];
    }

    inline T& at(size_t x, size_t y) {
        return this->pixels_[y * this->width_ + x];
    }

    inline const T& at(size_t x, size_t y) const {
        return this->pixels_[y * this->width_ + x];
    }

    inline T* data(size_t index = 0) {
        return this->pixels_.data() + index;
    }

    inline size_t width() const {
        return width_;
    }

    inline size_t height() const {
        return height_;
    }

    inline size_t size() const {
        return this->pixels_.size();
    }

    inline typename std::vector<T>::iterator begin() {
        return this->pixels_.begin();
    }

    inline typename std::vector<T>::iterator end() {
        return this->pixels_.end();
    }

    inline typename std::vector<T>::const_iterator cbegin() const noexcept {
        return this->pixels_.cbegin();
    }

    inline typename std::vector<T>::const_iterator cend() const noexcept {
        return this->pixels_.cend();
    }

    inline typename std::vector<T>::const_reverse_iterator crbegin() const noexcept {
        return this->pixels_.crbegin();
    }

    inline typename std::vector<T>::const_reverse_iterator crend() const noexcept {
        return this->pixels_.crend();
    }

    friend std::ostream& operator<<(std::ostream& os, const Image& image) {
        if (image.size() == 0) {
            std::cout << std::endl;
            return os;
        }

        size_t zero_fill = static_cast<size_t>(std::ceil(std::log10(image.size())));
        size_t digit_fill = static_cast<size_t>(std::max(static_cast<double>(zero_fill), 3.0));
        std::stringstream ss;

        for (size_t i = 0; i < image.size(); ++i) {
            T pixel = image[i];
            ss << std::setfill(' ') << std::setw(digit_fill);
            if (pixel != Image<T>::infinity) {
                ss << +pixel;
            } else {
                ss << "inf";
            }
            ss << " (" << std::setfill('0') << std::setw(zero_fill) << i << ") ";
            if ((i + 1) % image.width_ == 0 and i < image.size() - 1) {
                ss << std::endl;
            }
        }

        return os << ss.str();
    }

    std::vector<hsize_t> read(const std::string& path, const std::string& dataset_name, MPI_Comm comm) {
        HDF5File file(path, H5F_ACC_RDONLY);
        HDF5Dataset dataset(file, dataset_name);

        if (dataset.n_dims != 2) {
            std::stringstream message;
            message << "Image needs to be 2D, but is " << dataset.n_dims << "D" << std::endl;
            throw message.str();
        }

        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        hsize_t total_lines = dataset.dims[0];
        if (total_lines < static_cast<hsize_t>(size)) {
            std::stringstream message;
            message << "Number of lines in image needs to be greater than processing cores - lines: " << total_lines << " cores: " << size << std::endl;
            throw message.str();
        }
        hsize_t lines = dataset.dims[0] / size;
        hsize_t remainder = dataset.dims[0] % size;
        std::vector<hsize_t> counts = dataset.dims;
        counts[0] = lines + (static_cast<hsize_t>(rank) < remainder ? 1 : 0) +
                    (static_cast<hsize_t>(rank + 1) < std::min(static_cast<hsize_t>(size), total_lines) ? 1 : 0);
        std::vector<hsize_t> offsets(dataset.n_dims, 0);
        offsets[0] = lines * rank + std::min<hsize_t>(rank, remainder);

        this->height_ = counts[0];
        this->width_ = counts[1];
        this->pixels_.resize(counts[0] * counts[1]);
        dataset.read_chunks(this->pixels_.data(), counts.data(), offsets.data());

        return dataset.dims;
    }

    void write(const std::string& path, const std::string& dataset_name, MPI_Comm comm, std::vector<hsize_t>& dims) {
        int rank, size, message;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        hsize_t lines = dims[0] / size;
        hsize_t remainder = dims[0] % size;
        hsize_t counts[] = {(rank + 1 == size ? this->height_ : this->height_ - 1), this->width_};
        hsize_t offsets[] = {lines * rank + std::min<hsize_t>(rank, remainder), 0};

        // TODO: use mpiio

        if (rank == 0) {
            unsigned flags = !std::ifstream(path) ? H5F_ACC_EXCL : H5F_ACC_RDWR;
            {
                HDF5File file(path, flags);
                HDF5Dataset dataset(file, dataset_name);
                dataset.write_chunks(this->pixels_.data(), static_cast<int>(dims.size()), dims.data(), counts, offsets);
            }
            if (size > 1) {
                MPI_Send(&rank, 1, MPI_INT, 1, 0, comm);
            }
        } else {
            MPI_Recv(&message, 1, MPI_INT, rank - 1, 0, comm, MPI_STATUS_IGNORE);
            {
                HDF5File file(path, H5F_ACC_RDWR);
                HDF5Dataset dataset(file, dataset_name);
                dataset.write_chunks(this->pixels_.data(), static_cast<int>(dims.size()), dims.data(), counts, offsets);
            }
            if (rank + 1 < size) {
                MPI_Send(&rank, 1, MPI_INT, rank + 1, 0, comm);
            }
        }
    }
};

template<typename T>
const T Image<T>::infinity = std::numeric_limits<T>::max();

typedef Image<uint64_t> Parents;

#endif // IMAGE_H
