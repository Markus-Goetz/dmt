#ifndef DMT_MAX_TREE_H
#define DMT_MAX_TREE_H

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <unordered_map>
#include <vector>

#include "barrier.h"
#include "image.h"
#include "rules.h"

class MaxTree {
public:
    template <typename T, typename U=Parents::type>
    static void compute(const Image<T>& image, Parents& parents, unsigned int thread_count=std::thread::hardware_concurrency()) {
        Image<uint8_t> deja_vu(0, image.width(), image.height());

        // early out for a single core
        if (thread_count <= 1) {
            MaxTree::compute_chunk<T, U>(image, parents, deja_vu, 0ul, image.size());
            return;
        }

        thread_count = std::min(thread_count, static_cast<unsigned int>(image.height()));
        Barrier barrier(thread_count);
        std::vector<std::thread> threads;
        std::vector<AreaRules<U>> area_rules(thread_count);
//        std::cout << image << std::endl;
        for (unsigned int t = 0; t < thread_count; ++t) {
            threads.push_back(std::thread([t, thread_count, &barrier, &image, &parents, &deja_vu, &area_rules] {
                MaxTree::compute_threaded<T, U>(t, thread_count, barrier, image, parents, deja_vu, area_rules);
            }));
        }
        for (auto& thread : threads) thread.join();
    }

    // canonize a pixel according to the area remapping rules
    template<typename U=Parents::type>
    static U canonize(AreaRules<U>& rules, U origin) {
        U destination = origin;
        auto it = rules.find(destination);
        size_t i = 0;

        while (it != rules.end() and it->second != destination) {
            destination = it->second;
            it = rules.find(destination);
            ++i;
        }
        if (i > 1) rules[origin] = destination;

        return destination;
    }

    // determine the canonical point of a pixel
    template <typename T, typename U>
    static U canonical_point(U position, const Image<T>& image, Parents& parents) {
        T color = image[position];
        U parent = parents[position];

        return image[parent] == color ? parent : position;
    }

protected:
    // single neighbor
    template <typename T, typename U=Parents::type>
    struct Neighbor {
        U position;
        T color;
        Neighbor(T color_, U position_) : position(position_), color(color_) {}
    };

    template <typename T, typename U=Parents::type>
    struct Child {
        U canonical;
        T color;
        Child(T color_, U canonical_) : canonical(canonical_), color(color_) {}
    };

    // set of neighbors
    template <typename T, typename U=Parents::type>
    using Neighbors = std::vector<Neighbor<T, U>>;

    // get the right and lower neighbor
    template<typename T, typename U=Parents::type>
    static inline Neighbors<T> lower_neighbors(const Image<T>& image, U position, U end) {
        Neighbors<T, U> neighbors;

        // down
        if (position + image.width() < end) {
            size_t down = position + image.width();
            neighbors.push_back(Neighbor<T>(image[down], down));
        }

        // right
        if ((position + 1) % image.width() != 0 and position + 1 < end) {
            size_t right = position + 1;
            neighbors.push_back(Neighbor<T>(image[right], right));
        }

        return neighbors;
    }

    // get all 4-connected neighbors
    template<typename T, typename U=Parents::type>
    static inline Neighbors<T, U> all_neighbors(const Image<T>& image, U position, U start, U end) {
        Neighbors<T> neighbors = MaxTree::lower_neighbors(image, position, end);

        // left
        if (position % image.width() != 0 and position > start) {
            size_t left = position - 1;
            neighbors.push_back(Neighbor<T, U>(image[left], left));
        }

        // top
        if (position >= start + image.width()) {
            size_t top = position - image.width();
            neighbors.push_back(Neighbor<T, U>(image[top], top));
        }

        return neighbors;
    }

    template <typename T, typename U=Parents::type>
    static void compute_threaded(unsigned int thread, unsigned thread_count, Barrier& barrier, const Image<T>& image, Parents& parents, Image<uint8_t>& deja_vu, std::vector<AreaRules<U>>& area_rules) {
        // set thread affinity
        cpu_set_t cpu_set;
        CPU_ZERO(&cpu_set); CPU_SET(thread, &cpu_set);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set);

        // partition offsets
        U chunk = image.size() / thread_count;
        U remainder = image.size() % thread_count;

        // processing
        U start = thread * chunk + std::min(static_cast<U>(thread), remainder);
        U end = (thread + 1) * chunk + std::min(static_cast<U>(thread + 1), remainder);
        MaxTree::compute_chunk<T, U>(image, parents, deja_vu, start, end);
        barrier.wait();

        // merge partial results
        size_t steps = static_cast<size_t>(std::ceil(std::log2(thread_count)));
        for (size_t s = 0; s < steps; ++s) {
            size_t offset = 1 << s;
            size_t skip = 2 * offset;

            if (offset <= thread and (thread - offset) % skip == 0) {
                U rules_offset = std::max(1ul, offset / 2);
                std::swap(area_rules[thread], area_rules[thread - rules_offset]);
                auto& rules = area_rules[thread];
                auto& merge_rules = s > 0 ? area_rules[std::min<size_t>(thread + rules_offset, thread_count - 1)] : rules;
                MaxTree::merge_parents(image, parents, start, rules, merge_rules);
            }
            barrier.wait();
        }

        // apply the rules
        size_t access_point = 1 << (steps - 1);
        auto& rules = area_rules[access_point];
        MaxTree::apply_rules(parents, rules, start, end);
        barrier.wait();
    }

    // compute ax tree on a chunk of the image, implements salembier's algorithm
    template <typename T, typename U=Parents::type>
    static void compute_chunk(const Image<T>& image, Parents& parents, Image<uint8_t>& deja_vu, U start, U end) {
        std::map<T, std::vector<U> > stacks;
        std::unordered_map<T, std::vector<U> > pixels;
        std::unordered_map<U, U> area_values;
        std::vector<Child<T, U>> children;

        T start_color = image[start];
        stacks[start_color].push_back(start);
        pixels[start_color].push_back(start);
        deja_vu[start] = 1;

        while (!stacks.empty()) {
            flood:
            auto it = stacks.rbegin();
            T color = it->first;
            auto& bucket = it->second;
            U pixel = bucket.back();
            bucket.pop_back();

            Neighbors<T, U> neighbors = MaxTree::all_neighbors(image, pixel, start, end);
            for (const auto& neighbor : neighbors) {
                uint8_t& is_processed = deja_vu[neighbor.position];
                if (is_processed) continue;
                is_processed = 1;

                // add neighbors to queue
                pixels[neighbor.color].push_back(neighbor.position);
                stacks[neighbor.color].push_back(neighbor.position);

                // neighbor color is larger, descend first in depth-first-fashion
                if (color < neighbor.color) {
                    stacks[color].push_back(pixel);
                    goto flood;
                }
            }

            // the stack is empty, the current flat zone is found
            if (bucket.empty()) {
                auto& pixel_bucket = pixels[color];

                // determine canonical point
                std::sort(pixel_bucket.begin(), pixel_bucket.end());
                const U canonical = pixel_bucket.front();
                U area_size = pixel_bucket.size();

                // canonize flooded area
                for (auto& pixel : pixel_bucket) {
                    parents[pixel] = canonical;
                }

                pixels.erase(color);
                stacks.erase(--stacks.rbegin().base());

                // merge children if present
                T priority_color = !stacks.empty() ? stacks.rbegin()->first : color;
                while (!children.empty() and children.back().color > priority_color) {
                    parents[children.back().canonical] = canonical;
                    children.pop_back();
                }

                // add this area as child
                area_values[canonical] = area_size;
                children.push_back(Child<T, U>(priority_color, canonical));
            }
        }

        // connect remaining children to the root
        if (!children.empty()) {
            U root = children.back().canonical;
            for (auto& child : children) parents[child.canonical] = root;
        }
    }

    // merge the max trees of two neighboring images
    template <typename T, typename U>
    static void merge_parents(const Image<T>& image, Parents& parents, U start, AreaRules<U>& area, AreaRules<U>& merge_rules) {
        // merge previous area maps if needed
        for (auto& rule : merge_rules) {
            U left = MaxTree::canonize(area, rule.first);
            U right = MaxTree::canonize(merge_rules, rule.first);
            auto minmax = std::minmax(left, right);
            if (minmax.first == minmax.second) continue;

            area[rule.first] = minmax.first;
            area[minmax.second] = minmax.first;
        }
        if (area != merge_rules) merge_rules.clear();

        // detect new rules
        for (size_t i = start - image.width(), j = start; i < start; ++i, ++j) {
            MaxTree::merge_single(image, parents, area, i, j);
        }
        if (start % image.width() != 0) {
            MaxTree::merge_single(image, parents, area, start - 1, start);
        }
    }

    // merge a single pixel of the neighboring max trees
    template <typename T, typename U=Parents::type>
    static void inline merge_single(const Image<T>& image, Parents& parents, AreaRules<U>& area, size_t i, size_t j) {
        U x = MaxTree::canonize(area, MaxTree::canonical_point(j, image, parents));
        U y = MaxTree::canonize(area, MaxTree::canonical_point(i, image, parents));

        if (image[y] > image[x] or (image[y] == image[x] and y > x)) std::swap(x, y);
        while (x != y and y != Parents::infinity) {
            U z = parents[x] == x ? Parents::infinity : parents[x];
            if (image[z] >= image[y] and z != Parents::infinity) {
                x = z;
            } else if (image[x] == image[y]) {
                U x_canonized = MaxTree::canonize(area, x);
                U y_canonized = MaxTree::canonize(area, y);
                auto directed = std::minmax(x_canonized, y_canonized);
                area[directed.second] = directed.first;

                x = directed.first;
                y = MaxTree::closer(directed.first, directed.second, image, parents);
                if (y != Parents::infinity) parents[directed.first] = (y == z) ? parents[y_canonized] : z;
                parents[directed.second] = directed.first;
            } else {
                parents[x] = y;
                x = y;
                y = z;
            }
        }
    }

    template <typename T, typename U=Parents::type>
    static U closer(U x, U y, const Image<T>& image, Parents& parents) {
        U x_parent = parents[x];
        U y_parent = parents[y];
        if (x == x_parent or y == y_parent) return Parents::infinity;
        if (image[x_parent] > image[y_parent]) return x_parent;
        if (image[y_parent] > image[x_parent]) return y_parent;
        return std::max(x_parent, y_parent);
    }

    // remap the found area rules
    template <typename U>
    static void apply_rules(Parents& parents, AreaRules<U>& area, U start, U end) {
        for (size_t i = start; i < end; ++i) {
            U& pixel = parents[i];
            pixel = MaxTree::canonize(area, pixel);
        }
    }
};

#endif //DMT_MAX_TREE_H
