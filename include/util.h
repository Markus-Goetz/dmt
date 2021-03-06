#ifndef DMT_UTIL_H
#define DMT_UTIL_H

#include <iostream>
#include <map>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "image.h"
#include "tuple.h"

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    size_t i = 1;
    std::stringstream ss;
    size_t size = v.size();

    ss << "[";
    for (const auto& element : v) {
        ss << element;
        if (i < size) {
            ss << ", ";
        }
        ++i;
    }
    ss << "]";
    return os << ss.str();
}

template<typename T, typename U>
std::ostream& operator<<(std::ostream& os, const std::map<T, U>& m) {
    std::stringstream ss;
    ss << "{";
    if (m.size() > 0) {
        ss << std::endl;
    }
    for (auto& element : m) {
        ss << "\t" << element << "," << std::endl;
    }
    ss << "}";

    return os << ss.str();
}

template<typename T, typename U>
std::ostream& operator<<(std::ostream& os, const std::unordered_map<T, U>& m) {
    std::stringstream ss;
    ss << "{";
    if (m.size() > 0) {
        ss << std::endl;
    }
    for (auto& element : m) {
        ss << "\t" << element << "," << std::endl;
    }
    ss << "}";

    return os << ss.str();
}

template<typename T, typename U>
std::ostream& operator<<(std::ostream& os, const std::pair<T, U>& p) {
    std::stringstream ss;
    ss << (+p.first) << ": " << (p.second);
    return os << ss.str();
}

template <typename T, typename U=Parents::type>
std::ostream& operator<<(std::ostream& os, const TupleBuckets<T, U>& v) {
    std::stringstream ss;
    ss << "[" << std::endl;
    for (const auto& element : v) {
        if (element.second.empty()) continue;
        ss << "\t" << element << std::endl;
    }
    ss << "]";

    return os << ss.str();
}

#endif //DMT_UTIL_H
