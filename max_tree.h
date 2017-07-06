#ifndef DMT_MAX_TREE_H
#define DMT_MAX_TREE_H

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <unordered_map>
#include <vector>

#include "image.h"
#include "rules.h"
#include "thread_pool.h"

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

        // allocate resources
        thread_count = std::min(thread_count, static_cast<unsigned int>(image.height()));
        std::vector<AreaRules<U>> area_rules(thread_count);
        ThreadPool pool(thread_count);

        // partition offsets
        U chunk = image.size() / thread_count;
        U remainder = image.size() % thread_count;

        // processing loop
        for (size_t t = 0; t < thread_count; ++t) {
            U start = t * chunk + std::min(t, remainder);
            U end = (t + 1) * chunk + std::min(t + 1, remainder);
            pool.add_job([&image, &parents, &deja_vu, start, end] {
                MaxTree::compute_chunk<T, U>(image, parents, deja_vu, start, end);
            });
        }
        pool.wait_all();

        // merge partial results
        size_t steps = static_cast<size_t>(std::ceil(std::log2(thread_count)));
        for (size_t s = 0; s < steps; ++s) {
            size_t offset = static_cast<size_t>(std::pow(2.0, s));
            size_t skip = 2 * offset;

            for (size_t t = 0; t < thread_count; t += skip) {
                U merge_point = t + offset;
                U start = merge_point * chunk + std::min(merge_point, remainder);
                if (start >= image.size() or merge_point >= thread_count) continue;

                auto& rules = area_rules[t];
                auto& merge_rules = area_rules[merge_point];

                pool.add_job([&image, &parents, start, &rules, &merge_rules] {
                    MaxTree::merge_parents(image, parents, start, rules, merge_rules);
                });
            }
            pool.wait_all();
        }

        // apply the rules
        auto& rules = area_rules.front();
        for (size_t t = 0; t < thread_count; ++t) {
            U start = t * chunk + std::min(t, remainder);
            U end = (t + 1) * chunk + std::min(t + 1, remainder);
            pool.add_job([&parents, &rules, start, end] {
                MaxTree::apply_rules(parents, rules, start, end);
            });
        }
        pool.join_all();
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

    // compute ax tree on a chunk of the image, implements salembier's algorithm
    template <typename T, typename U=Parents::type>
    static void compute_chunk(const Image<T>& image, Parents& parents, Image<uint8_t>& deja_vu, U start, U end) {
        std::map<T, std::vector<U> > stacks;
        std::unordered_map<T, std::vector<U> > pixels;
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
                stacks[neighbor.color].push_back(neighbor.position);
                auto& pixel_bucket = pixels[neighbor.color];
                pixel_bucket.push_back(neighbor.position);
                if (pixel_bucket.front() > pixel_bucket.back()) {
                    std::swap(pixel_bucket.front(), pixel_bucket.back());
                }


                // neighbor color is larger, descend first in depth-first-fashion
                if (color < neighbor.color) {
                    stacks[color].push_back(pixel);
                    goto flood;
                }
            }

            // the stack is empty, the current flat zone is found
            if (bucket.empty()) {
                // determine canonical point
                auto& pixel_bucket = pixels[color];
                const U canonical = pixel_bucket.front();

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
            if (left == right) continue;
            auto minmax = std::minmax(left, right);

            area[rule.first] = minmax.first;
            area[minmax.second] = minmax.first;
        }
        merge_rules.clear();

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
            if (z != Parents::infinity and image[z] >= image[y]) {
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
        if (x == x_parent) { if (y == y_parent) return Parents::infinity; else return y_parent; }
        if (y == y_parent) return Parents::infinity;
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
