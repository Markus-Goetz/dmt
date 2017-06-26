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
        for (size_t t = 0; t < thread_count; ++t) {
            U start = t * chunk + std::min(t, remainder);
            U end = (t + 1) * chunk + std::min(t + 1, remainder);
            auto& rules = area_rules.front();
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

        while (it != rules.end() and it->second != destination) {
            destination = it->second;
            it = rules.find(destination);
        }
        if (origin != destination) rules[origin] = destination;

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
        std::vector<T> merge_color;
        std::vector<U> children;
        std::map<T, std::vector<U> > stacks;
        std::unordered_map<T, std::vector<U> > pixels;

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

                // canonize flooded area
                for (auto& pixel : pixel_bucket) {
                    parents[pixel] = canonical;
                }

                pixels.erase(color);
                stacks.erase(--stacks.rbegin().base());

                // merge children if present
                T priority_color = !stacks.empty() ? stacks.rbegin()->first : color;
                while (!children.empty() and merge_color.back() > priority_color) {
                    parents[children.back()] = canonical;
                    children.pop_back();
                    merge_color.pop_back();
                }

                // add this area as child
                children.push_back(canonical);
                merge_color.push_back(priority_color);
            }
        }

        // connect remaining children to the root
        if (!children.empty()) {
            U root = children.back();
            for (auto& child : children) parents[child] = root;
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

    template <typename T, typename U=Parents::type>
    static void get_chain(const Image<T>& image, Parents& parents, U start, std::vector<std::pair<T, U>>& chain) {
        U current = start;
        while (true) {
            chain.push_back(std::pair<T, U>(image[current], current));
            if (parents[current] == current) break;
            current = parents[current];
        }
    }

    // merge a single pixel of the neighboring max trees
    template <typename T, typename U=Parents::type>
    static void merge_single(const Image<T>& image, Parents& parents, AreaRules<U>& area, size_t i, size_t j) {
        std::vector<std::pair<T, U>> chain;
        U left = MaxTree::canonize(area, MaxTree::canonical_point(i, image, parents));
        U right = MaxTree::canonize(area, MaxTree::canonical_point(j, image, parents));

        MaxTree::get_chain<T, U>(image, parents, left, chain);
        size_t middle = chain.size();
        MaxTree::get_chain<T, U>(image, parents, right, chain);
        std::inplace_merge(chain.begin(), chain.begin() + middle, chain.end(), std::greater<std::pair<T, U>>());

        for (auto it = chain.begin(); it != chain.end() - 1; ++it) {
            auto next = it + 1;

            // merge areas
            if (it->first == next->first) {
                auto directed = std::minmax(it->second, next->second);
                if (directed.first == directed.second) break;
                area[directed.second] = directed.first;
                parents[directed.second] = directed.first;
                next->second = directed.first;
            } else {
                parents[it->second] = next->second;
            }
        }
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
