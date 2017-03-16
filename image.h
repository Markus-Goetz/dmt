#ifndef IMAGE_H
#define IMAGE_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

template<typename T>
class Image {
protected:
    std::vector<T> pixels_;

public:
    static const T infinity;

    size_t width_;
    size_t height_;

    typedef T type;

    Image()
            : width_(0), height_(0) {}

    Image(std::initializer_list<T> pixels, size_t width, size_t height)
            : pixels_(pixels), width_(width), height_(height) {
        assert(this->pixels_.size() == this->width_ * this->height_);
    }

    Image(size_t width, size_t height)
            : pixels_(width * height), width_(width), height_(height) {}

    Image(T fill, size_t width, size_t height)
            : pixels_(width * height, fill), width_(width), height_(height) {}

    inline T& operator[](size_t index) {
        return this->pixels_[index];
    }

    inline const T& operator[](size_t index) const {
        return this->pixels_[index];
    }

    inline T& at(size_t x, size_t y) {
        return this->pixels_[y * width + x];
    }

    inline const T& at(size_t x, size_t y) const {
        return this->pixels_[y * width + x];
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

    friend std::ostream& operator<<(std::ostream& os, const Image& image) {
        const T& maximum = *std::max_element(image.cbegin(), image.cend());

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

    /*
     *     // Open the HDF5 file and the dataset DBSCAN in it
    try
    {
        hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        hid_t dset = H5Dopen1(file, DATASET);
        hid_t fileSpace= H5Dget_space(dset);

        // Read dataset size and calculate chunk size
        hsize_t count[2];
        H5Sget_simple_extent_dims(fileSpace, count,NULL);
        this->m_totalSize = count[0];
        hsize_t chunkSize =(this->m_totalSize / this->m_mpiSize) + 1;
        hsize_t offset[2] = {this->m_mpiRank * chunkSize, 0};
        count[0] = std::min(chunkSize, this->m_totalSize - offset[0]);

        // Initialize members
        this->m_size         = count[0];
        this->m_dimensions   = count[1];
        this->m_cells        = new Cell[this->m_size];
        this->m_points       = new Coord[this->m_size * this->m_dimensions];
        this->m_initialOrder = new size_t[this->m_size];
        std::iota(this->m_initialOrder, this->m_initialOrder + this->m_size, this->m_mpiRank * chunkSize);

        // Read data
        hid_t memSpace = H5Screate_simple(2, count, NULL);
        H5Sselect_hyperslab(fileSpace,H5S_SELECT_SET,offset, NULL, count, NULL);
        H5Dread(dset, H5T_IEEE_F32LE, memSpace, fileSpace,H5P_DEFAULT, m_points);

        // Check if there is an "Cluster" dataset in the file
        if (!this->m_mpiRank)
        {
            htri_t exists = H5Lexists(file, "Clusters", H5P_DEFAULT);
            if (!exists)
            {
                hsize_t dims[2] = {this->m_totalSize, 1};
                hid_t globalSpace = H5Screate_simple(1,dims,NULL);
                hid_t clusterSet = H5Dcreate1(file, "Clusters", H5T_NATIVE_LONG ,globalSpace, H5P_DEFAULT);
                H5Fclose(clusterSet);
            }
        }

        // Close file and dataset
        H5Dclose(dset);
        H5Fclose(file);
    }
    catch(herr_t error)
    {
        if (!this->m_mpiRank)
        {
            std::cerr << "Could not open file " << filename << std::endl;
        }
        exit(this->m_mpiRank ? EXIT_SUCCESS : EXIT_FAILURE);
     *
     */
};

template<typename T>
const T Image<T>::infinity = std::numeric_limits<T>::max();

typedef Image<uint64_t> Parents;

#endif // IMAGE_H

