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
};

template<typename T>
const T Image<T>::infinity = std::numeric_limits<T>::max();

typedef Image<uint64_t> Parents;

#endif // IMAGE_H

