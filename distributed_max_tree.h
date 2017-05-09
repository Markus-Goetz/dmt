#ifndef DISTRIBUTED_MAX_TREE_H
#define DISTRIBUTED_MAX_TREE_H

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <iostream>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <mpi.h>

#include "endpoints.h"
#include "image.h"
#include "tuple.h"
#include "mpi_wrapper.h"
#include "util.h"

template <typename T=Parents::type>
using Bucket = std::vector<T>;

template <typename T=Parents::type>
using Buckets = std::vector<Bucket<T>>;

template <typename T, typename U=Parents::type>
using Neighbor = std::tuple<T, U>;

template <typename T, typename U=Parents::type>
using Neighbors = std::vector<Neighbor<T, U>>;

template <typename T=Parents::type>
using AreaRules = std::unordered_map<T, T>;

template <typename T, typename U=Parents::type>
using Root = std::pair<T, U>;

template <typename T, typename U=Parents::type>
using RootRules = std::unordered_map<U, Root<T, U>>;

template <typename T, typename U=Parents::type>
std::ostream& operator<<(std::ostream& os, const TupleBuckets<T, U>& v) {
    std::stringstream ss;
    ss << "[" << std::endl;
    for (const auto& element : v) {
        if (element.empty()) continue;
        ss << "\t" << element << std::endl;
    }
    ss << "]";

    return os << ss.str();
}

class DistributedMaxTree {
public:
    DistributedMaxTree(MPI_Comm comm = MPI_COMM_WORLD)
            : comm_(comm) {
        MPI_Comm_rank(comm, &this->rank_);
        MPI_Comm_size(comm, &this->size_);
    }

    template<typename T, typename U=Parents::type>
    Parents compute(const Image<T>& image) {
        AreaRules<U> area;

        // create the reverse communicator used for the stitching scans
        MPI_Comm_split(this->comm_, 0, this->size_ - this->rank_ - 1, &this->reverse_comm_);

        // calculate the global pixel offset
        size_t offset = image.size() - image.width();
        MPI_Exscan(MPI_IN_PLACE, &offset, 1, MPI_Types<U>::map(), MPI_SUM, this->comm_);
        offset = this->rank_ == 0 ? 0 : offset;

        // allocate the parent image and buckets for the tuples
        Parents parents(Parents::infinity, image.width(), image.height());

        // MPI type and op creation
        Tuple<T, U>::create_mpi_type(&this->tuple_type_);
        AreaEndpoint<U>::create_mpi_type(&this->area_endpoint_type_);
        Endpoint<T, U>::create_mpi_type(&this->endpoint_type_);

        this->create_right_stitch_op<U>(parents, area);
        this->create_left_stitch_op<U>(parents, area);
        this->create_area_connect_op<U>();
//        this->create_level_connect_op<T, U>();

        // here is the meat!
        TupleBuckets<T, U> root_buckets = this->get_local_tuples(image, parents, offset);
        TupleBuckets<T, U> area_buckets = this->connect_halos(image, parents, offset, area, root_buckets.size());
        this->resolve_tuples(area_buckets, root_buckets);
//        Tuples<T, U> resolved_tuples = this->redistribute_tuples(image, tuple_buckets, resolved_tuples);
//        this->generate_parent_image(parents, resolved_tuples, area, offset);

        // MPI clean up
        MPI_Op_free(&this->area_connect_);
//        MPI_Op_free(&this->level_connect_);
        MPI_Op_free(&this->left_stitch_);
        MPI_Op_free(&this->right_stitch_);

        MPI_Type_free(&this->endpoint_type_);
        MPI_Type_free(&this->area_endpoint_type_);
        MPI_Type_free(&this->tuple_type_);

        MPI_Comm_free(&this->reverse_comm_);

        return parents;
    }

protected:
    MPI_Comm comm_;
    MPI_Comm reverse_comm_;

    MPI_Datatype tuple_type_;
    MPI_Datatype area_endpoint_type_;
    MPI_Datatype endpoint_type_;

    MPI_Op right_stitch_;
    MPI_Op left_stitch_;
    MPI_Op area_connect_;
//    MPI_Op level_connect_;

    int rank_;
    int size_;

    template <typename T, typename U=Parents::type>
    void radix_sort(Tuples<T, U>& tuples) {
        const size_t bytes = sizeof(Tuple<T, U>);
        uint8_t buffer[Tuple<T, U>::bytes];

        std::vector<size_t> histogram(bytes * (std::numeric_limits<uint8_t>::max() + 1));
        std::vector<size_t> displacement(bytes * (std::numeric_limits<uint8_t>::max() + 1));

        // calculate histogram
        for (auto& tuple : tuples) {
            tuple.pack(buffer);
            for (size_t i = bytes - 1; i != std::numeric_limits<size_t>::max(); --i) {
                ++histogram[i * (std::numeric_limits<uint8_t>::max() + 1) + buffer[i]];
            }
        }

        // calculate offsets
        for (size_t i = 0; i < bytes; ++i) {
            for (size_t j = 1; j < (std::numeric_limits<uint8_t>::max() + 1); ++j) {
                displacement[i * (std::numeric_limits<uint8_t>::max() + 1) + j] =
                        displacement[i * (std::numeric_limits<uint8_t>::max() + 1) + j - 1] +
                        histogram[i * (std::numeric_limits<uint8_t>::max() + 1) + j - 1];
            }
        }

        // radix sort
        Tuples<T, U> output(tuples.size());
        for (size_t i = bytes - 1; i != std::numeric_limits<size_t>::max(); --i) {
            for (auto& tuple : tuples) {
                tuple.pack(buffer);
                size_t& target = displacement[i * (std::numeric_limits<uint8_t>::max() + 1) + buffer[i]];
                output[target] = tuple;
                ++target;
            }
            tuples.swap(output);
        }
        tuples.swap(output);
    };

    template<typename U=Parents::type>
    void create_right_stitch_op(const Parents& parents, AreaRules<U>& area) {
        auto right_rule_scan = [this, &parents, &area](void* in_, void* out_, int* len, MPI_Datatype*) {
            U* in = reinterpret_cast<U*>(in_);
            U* out = reinterpret_cast<U*>(out_);

            for (int i = 0; i < *len; ++i) {
                auto min_max = std::minmax(in[i], parents[i]);
                U remote = min_max.first;
                U local = min_max.second;
                auto it = area.find(local);

                if (it != area.end() and it->second != remote) {
                    area[local] = std::min(this->canonize(area, remote), this->canonize(area, it->second));
                } else if (local != remote) {
                    area[local] = remote;
                }
            }

            size_t offset = parents.width() * parents.height() - parents.width();
            for (int i = 0; i < *len; ++i) {
                out[i] = this->canonize(area, parents[offset + i]);
            }
        };
        static auto static_right_scan = right_rule_scan;

        MPI_Op_create([](void* in, void* out, int* len, MPI_Datatype* type) {
            static_right_scan(in, out, len, type);
        }, false, &this->right_stitch_);
    }

    template<typename U=Parents::type>
    void create_left_stitch_op(const Parents& parents, AreaRules<U>& area) {
        auto left_rule_scan = [this, &parents, &area](void* in_, void* out_, int* len, MPI_Datatype*) {
            U* in = reinterpret_cast<U*>(in_);
            U* out = reinterpret_cast<U*>(out_);
            size_t offset = parents.width() * parents.height() - parents.width();

            for (int i = 0; i < *len; ++i) {
                auto min_max = std::minmax(in[i], parents[offset + i]);
                U remote = min_max.first;
                U local = min_max.second;

                if (local != remote) {
                    area[local] = remote;
                }
            }

            for (int i = 0; i < *len; ++i) {
                out[i] = this->canonize(area, parents[i]);
            }
        };
        static auto static_left_scan = left_rule_scan;

        MPI_Op_create([](void* in, void* out, int* len, MPI_Datatype* type) {
            static_left_scan(in, out, len, type);
        }, false, &this->left_stitch_);
    }

    template<typename U=Parents::type>
    void create_area_connect_op() {
        auto connect = [](void* in_, void* out_, int*, MPI_Datatype*) {
            AreaEndpoint<U>* in = reinterpret_cast<AreaEndpoint<U>*>(in_);
            AreaEndpoint<U>* out = reinterpret_cast<AreaEndpoint<U>*>(out_);

            AreaEndpoint<U>& out_left = out[0];
            AreaEndpoint<U>& out_right = out[1];
            AreaEndpoint<U>& in_right = in[1];

            if (in_right.from == out_left.from) {
                out_left.canonical_to = std::min(in_right.canonical_to, out_left.canonical_to);
            }

            // very long chain? just copy the already merged values
            if (out_left.from == out_right.from) {
                out_right = out_left;
            }
        };

        MPI_Op_create(connect, false, &this->area_connect_);
    }

//    template<typename T, typename U=Parents::type>
//    void create_level_connect_op() {
//        auto connect = [](void* in_, void* out_, int*, MPI_Datatype*) {
//            Endpoint<T, U>* in = reinterpret_cast<Endpoint<T, U>*>(in_);
//            Endpoint<T, U>* out = reinterpret_cast<Endpoint<T, U>*>(out_);
//
//            Endpoint<T, U>& out_left = out[0];
//            Endpoint<T, U>& out_right = out[1];
//            Endpoint<T, U>& in_right = in[1];
//
//            if (in_right.from == out_left.from) {
//                out_left.current_from = std::min(in_right.current_from, out_left.current_from);
//
//                if ((out_left.from == out_left.current_root_to) or // became a node
//                    (in_right.current_root_color > out_left.current_root_color) or // closer colored root
//                    (in_right.current_root_color == out_left.current_root_color and // same colored root, but ...
//                     in_right.current_root_to < out_left.current_root_to)) { // ... smaller canonical point
//                    out_left.current_root_color = in_right.current_root_color;
//                    out_left.current_root_to = in_right.current_root_to;
//                }
//            } else if (in_right.start_root_to == out_left.current_root_to and
//                       in_right.current_root_color == in_right.start_root_color) {
//                out_left.current_root_to = std::min(in_right.current_root_to, out_left.current_root_to);
//            }
//
//            // very long chain? just copy the already merged values
//            if (out_left.from == out_right.from) {
//                out_right = out_left;
//            }
//        };
//
//        MPI_Op_create(connect, false, &this->level_connect_);
//    }

    template<typename T, typename U=Parents::type>
    Neighbors<T> lower_neighbors(const Image<T>& image, U position) {
        Neighbors<T, U> neighbors;

        // right
        if ((position + 1) % image.width() != 0) {
            size_t right = position + 1;
            neighbors.push_back(Neighbor<T>(image[right], right));
        }

        // down
        if (position + image.width() < image.size()) {
            size_t down = position + image.width();
            neighbors.push_back(Neighbor<T>(image[down], down));
        }

        return neighbors;
    }

    template<typename T, typename U=Parents::type>
    Neighbors<T, U> all_neighbors(const Image<T>& image, size_t position) {
        Neighbors<T> neighbors = lower_neighbors(image, position);

        // top
        if (position >= image.width()) {
            size_t top = position - image.width();
            neighbors.push_back(Neighbor<T, U>(image[top], top));
        }
        // left
        if (position % image.width() != 0) {
            size_t left = position - 1;
            neighbors.push_back(Neighbor<T, U>(image[left], left));
        }

        return neighbors;
    }

    template<typename U=Parents::type>
    U find_root(const Parents& parents, U position, size_t offset) {
        U parent = parents[position];
        if (parent == position + offset) {
            return parent;
        }
        return find_root(parents, parent - offset, offset);
    }

    template<typename T, typename U=Parents::type>
    TupleBuckets<T, U> get_local_tuples(const Image<T>& image, Parents& parents, size_t offset) {
        // bucket sort image
        T max_color = 0;
        Buckets<U> buckets;
        for (size_t i = 0; i < image.size(); ++i) {
            T color = image[i];
            buckets.resize(std::max(static_cast<size_t>(color + 1), buckets.size()));
            buckets[color].push_back(i);
            max_color = std::max(color, max_color);
        }

        // get global max channel
        MPI_Allreduce(MPI_IN_PLACE, &max_color, 1, MPI_Types<T>::map(), MPI_MAX, this->comm_);
        buckets.resize(max_color + 1);
        TupleBuckets<T, U> root_buckets(max_color + 1);

        // connected component labeling
        std::map<U, std::set<U> > outbound;
        T color = static_cast<T>(buckets.size() - 1);
        for (auto bucket = buckets.crbegin(); bucket != buckets.crend(); ++bucket) {
            // weakly connect the sorted pixels to flat zones per gray-value
            for (auto index = bucket->crbegin(); index != bucket->crend(); ++index) {
                parents[*index] = *index + offset;
                Neighbors<T> neighbors = lower_neighbors(image, *index);
                for (auto neighbor : neighbors) {
                    T neighbor_color = std::get<0>(neighbor);
                    U neighbor_position = std::get<1>(neighbor);

                    if (neighbor_color != color) continue;
                    U root = find_root(parents, neighbor_position, offset);
                    parents[root - offset] = *index + offset;
                }
            }

            // strongly connect zones and generate tuples
            outbound.clear();
            for (auto index = bucket->cbegin(); index != bucket->cend(); ++index) {
                U root = find_root(parents, *index, offset);
                parents[*index] = root;

                Neighbors<T> neighbors = all_neighbors(image, *index);
                for (auto neighbor : neighbors) {
                    // skip lower color bands
                    T neighbor_color = std::get<0>(neighbor);
                    if (neighbor_color <= color) continue;

                    // get the root of the neighbor
                    std::set<U>& out = outbound[root];
                    U neighbor_position = std::get<1>(neighbor);
                    U neighbor_root = find_root(parents, neighbor_position, offset);

                    // generate connection tuples to already canonized zones
                    if (out.find(neighbor_root) == out.end()) {
                        out.insert(neighbor_root);
                        Tuples<T, U>& bucket = root_buckets[neighbor_color];
                        bucket.push_back(Tuple<T, U>(neighbor_color, neighbor_root, color, root));
                    }
                }
            }

            // advance to the next color channel
            --color;
        }

        return root_buckets;
    }

    template<typename U=Parents::type>
    U canonize(AreaRules<U>& rules, U origin) {
        U destination = origin;
        auto it = rules.find(destination);

        while (it != rules.end() and it->second != destination) {
            destination = it->second;
            it = rules.find(destination);
        }
        return destination;
    }

    template<typename T, typename U=Parents::type>
    TupleBuckets<T, U> connect_halos(const Image<T>& image, Parents& parents, size_t global_offset, AreaRules<U>& area, size_t max_color) {
        TupleBuckets<T, U> area_buckets(max_color);
        if (this->size_ == 1) return area_buckets;

        size_t width = parents.width();
        std::vector<U> send_buffer(width);
        std::vector<U> recv_buffer(width);
        size_t offset = width * parents.height() - width;

        // right directed communication of minimum
        U* read = parents.data() + (this->rank_ != 0 ? 0 : offset);
        MPI_Scan(read, recv_buffer.data(), width, MPI_Types<U>::map(), this->right_stitch_, this->comm_);

        // left directed, backwards propagation of right minimum scan results
        for (size_t i = 0; i < width; ++i) {
            send_buffer[i] = this->canonize(area, parents[i + (this->rank_ + 1 == this->size_ ? 0 : offset)]);
        }
        MPI_Scan(send_buffer.data(), recv_buffer.data(), width,
                 MPI_Types<U>::map(), this->left_stitch_, this->reverse_comm_);

        for (const auto& rule : area) {
            T color = image[rule.first - global_offset];
            area_buckets[color].push_back(Tuple<T, U>(color, rule.first, color, rule.second));
            area_buckets[color].push_back(Tuple<T, U>(color, rule.second, color, rule.first));
        }

        return area_buckets;
    }

    template<typename T, typename U=Parents::type>
    void resolve_tuples(TupleBuckets<T, U>& area_buckets, TupleBuckets<T, U>& root_buckets) {
        const size_t max_color = area_buckets.size() - 1;
        for (size_t i = max_color; i != std::numeric_limits<size_t>::max(); --i) {
            // retrieve the current bucket
            const T color = static_cast<T>(i);
            Tuples<T, U>& area_bucket = area_buckets[color];
            Tuples<T, U>& root_bucket = root_buckets[color];

            // check whether there are tuples in this channel at all, if none are available, skip the channel
            size_t total = area_bucket.size() + root_bucket.size();
            MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_Types<decltype(total)>::map(), MPI_SUM, this->comm_);
            if (total == 0) {
                continue;
            }

            // resolve the tuple chains
            bool unresolved= true;
            while (unresolved) {
                AreaRules<U> area;
                this->sample_sort(area_bucket);
                this->resolve_area_chain(area_bucket, area);
                unresolved = this->remap_area_tuples(area_bucket, area);
                MPI_Allreduce(MPI_IN_PLACE, &unresolved, 1, MPI_C_BOOL, MPI_LOR, this->comm_); // globally done?
            }

            // put the correct area tuple into the root bucket
            for (const auto& tuple : area_bucket) {
                if (tuple.from > tuple.to) {
                    root_bucket.push_back(tuple);
                }
            }
            this->resolve_roots(color, root_buckets, area_buckets);
        }
    }

    template<typename T, typename U=Parents::type>
    void sample_sort(Tuples<T, U>& bucket) {
        // early out for a single core
        if (this->size_ == 1) {
            std::stable_sort(bucket.begin(), bucket.end());
            bucket.erase(std::unique(bucket.begin(), bucket.end()), bucket.end());
            return;
        }

        // balance work load
        this->balance_tuples(bucket);

        // allocate space for the splitters and prepare a no-op dummy
        const size_t split_count = static_cast<size_t>(this->size_ - 1);
        size_t local_elements = bucket.size();
        Tuple<T, U> max_dummy;
        Tuples<T, U> splitters(this->size_ * split_count);

        // ... sort locally
        std::stable_sort(bucket.begin(), bucket.end());

        // ... and sample sort it globally
        size_t valid_splits = std::min(split_count, local_elements);
        double local_skip = static_cast<double>(local_elements) / static_cast<double>(this->size_);

        // exchange split points
        std::fill(splitters.begin(), splitters.end(), max_dummy);
        for (U split = 0; split < valid_splits; ++split) {
            splitters[split] = bucket[static_cast<size_t>(std::round((split + 1) * local_skip))];
        }
        MPI_Allreduce(MPI_IN_PLACE, &valid_splits, 1, MPI_Types<U>::map(), MPI_SUM, this->comm_);
        MPI_Allgather(MPI_IN_PLACE, static_cast<int>(split_count), this->tuple_type_,
                      splitters.data(), static_cast<int>(split_count), this->tuple_type_, this->comm_);

        // sort the split points
        std::stable_sort(splitters.begin(), splitters.end());

        // determine tuple destinations by linearly scanning the tuples and dividing them based on the splits
        size_t target_rank = 0;
        Tuple<T, U>& current_split = splitters[0];
        local_skip = valid_splits / static_cast<double>(this->size_);

        // distribute the tuples according to the chosen split points onto the mpi ranks from 0 to n
        std::vector<int> send_counts(this->size_, 0);
        std::vector<int> send_displs(this->size_, 0);
        std::vector<int> recv_counts(this->size_, 0);
        std::vector<int> recv_displs(this->size_, 0);

        for (const Tuple<T, U>& tuple : bucket) {
            while (tuple > current_split) {
                ++target_rank;
                if (target_rank == split_count) {
                    current_split = max_dummy;
                } else {
                    current_split = splitters[static_cast<size_t>(std::round((target_rank + 1) * local_skip))];
                }
            }
            ++send_counts[target_rank];
        }
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, this->comm_);

        // calculate the displacements for the all-to-all communication
        int total = recv_counts[0];
        for (int rank = 1; rank < this->size_; ++rank) {
            total += recv_counts[rank];
            send_displs[rank] = send_displs[rank - 1] + send_counts[rank - 1];
            recv_displs[rank] = recv_displs[rank - 1] + recv_counts[rank - 1];
        }

        // communicate the globally sorted tuples, sort them again locally and return the result
        Tuples<T, U> incoming_tuples(total);
        MPI_Alltoallv(
                bucket.data(), send_counts.data(), send_displs.data(), this->tuple_type_,
                incoming_tuples.data(), recv_counts.data(), recv_displs.data(), this->tuple_type_, this->comm_
        );

        // merge incoming already sorted sequences
        for (int i = 0; i < this->size_ - 1; ++i) {
            auto middle = incoming_tuples.begin() + recv_displs[i + 1];
            auto last = incoming_tuples.begin() + (i + 2 >= this->size_ ? static_cast<int>(incoming_tuples.size()) : recv_displs[i + 2]);
            std::inplace_merge(incoming_tuples.begin(), middle, last);
        }
        incoming_tuples.erase(std::unique(incoming_tuples.begin(), incoming_tuples.end()), incoming_tuples.end());

        // exchange the incoming tuples with the bucket
        bucket.swap(incoming_tuples);
        this->balance_tuples(bucket);
    };

    template <typename T, typename U=Parents::type>
    void balance_tuples(Tuples<T, U>& bucket) {
        size_t local_elements = bucket.size();
        size_t total_elements = local_elements;
        size_t left_elements = local_elements;

        MPI_Allreduce(MPI_IN_PLACE, &total_elements, 1, MPI_Types<decltype(total_elements)>::map(), MPI_SUM, this->comm_);
        MPI_Exscan(MPI_IN_PLACE, &left_elements, 1, MPI_Types<decltype(left_elements)>::map(), MPI_SUM, this->comm_);
        left_elements = this->rank_ != 0 ? left_elements : 0;

        size_t chunk = total_elements / this->size_;
        size_t remainder = total_elements % this->size_;

        int target_rank = (chunk == 0) ? static_cast<int>(left_elements) : static_cast<int>(left_elements / chunk);
        while (target_rank * chunk + std::min(static_cast<size_t>(target_rank), remainder) > left_elements) {
            --target_rank;
        }

        std::vector<int> send_counts(this->size_, 0);
        std::vector<int> recv_counts(this->size_, 0);
        while (local_elements > 0) {
            size_t end = (target_rank + 1) * chunk + std::min(static_cast<size_t>(target_rank + 1), remainder);
            send_counts[target_rank] = static_cast<int>(std::min(end - left_elements, local_elements));
            local_elements -= send_counts[target_rank];
            left_elements = end;
            ++target_rank;
        }
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, this->comm_);

        std::vector<int> send_displs(this->size_, 0);
        std::vector<int> recv_displs(this->size_, 0);
        for (int rank = 1; rank < this->size_; ++rank) {
            send_displs[rank] = send_displs[rank - 1] + send_counts[rank - 1];
            recv_displs[rank] = recv_displs[rank - 1] + recv_counts[rank - 1];
        }

        Tuples<T, U> incoming(chunk + (static_cast<size_t>(this->rank_) < remainder ? 1 : 0));
        MPI_Alltoallv(bucket.data(), send_counts.data(), send_displs.data(), this->tuple_type_,
                      incoming.data(), recv_counts.data(), recv_displs.data(), this->tuple_type_, this->comm_);
        bucket.swap(incoming);
    };

    template <typename T, typename U=Parents::type>
    void resolve_area_chain(Tuples<T, U>& tuples, AreaRules<U>& area) {
        for (const auto& tuple : tuples) {
            U from = tuple.from;
            U to = tuple.to;

            if (to > from) std::swap(from, to);
            U canonical_point = this->canonize(area, from);

            if (to == canonical_point) {
                area[from] = to;
            } else {
                auto targets = std::minmax(to, canonical_point);
                area[from] = targets.first;
                area[targets.second] = targets.first;
            }
        }

        if (this->size_ == 1) return;
        // tuple chain is locally resolved, exchange information globally
        AreaEndpoints<U> ends = this->initialize_area_ends(tuples, area);
        MPI_Scan(MPI_IN_PLACE, ends.data(), ends.size(), this->area_endpoint_type_, this->area_connect_, this->comm_);
        this->stitch_area_ends(ends, area);

        ends = this->initialize_area_ends(tuples, area);
        std::swap(ends.front(), ends.back());
        MPI_Scan(MPI_IN_PLACE, ends.data(), ends.size(), this->area_endpoint_type_, this->area_connect_, this->reverse_comm_);
        std::swap(ends.front(), ends.back());
        this->stitch_area_ends(ends, area);
    };


    template <typename T, typename U=Parents::type>
    AreaEndpoints<U> initialize_area_ends(Tuples<T, U>& tuples, AreaRules<U>& area) {
        AreaEndpoints<U> ends;
        if (tuples.empty()) return ends;

        Tuple<T, U>& front = tuples.front();
        AreaEndpoint<U>& left = ends.front();
        auto left_from_to = std::minmax(front.from, front.to);
        left.from = left_from_to.second;
        left.to = left_from_to.first;
        left.canonical_to = this->canonize(area, left.from);

        Tuple<T, U>& back = tuples.back();
        AreaEndpoint<U>& right = ends.back();
        auto right_from_to = std::minmax(back.from, back.to);
        right.from = right_from_to.second;
        right.to = right_from_to.first;
        right.canonical_to = this->canonize(area, right.from);

        return ends;
    };

    template <typename U=Parents::type>
    void stitch_area_ends(AreaEndpoints<U>& ends, AreaRules<U>& area) {
        const AreaEndpoint<U>& left = ends.front();
        const AreaEndpoint<U>& right = ends.back();

        if (left.to != left.canonical_to)
            area[left.to] = left.canonical_to;
        if (right.to != right.canonical_to)
            area[right.to] = right.canonical_to;
    };

    template <typename T, typename U=Parents::type>
    bool remap_area_tuples(Tuples<T, U>& area_bucket, AreaRules<U>& area) {
        Tuples<T, U> bucket;
        bucket.swap(area_bucket);

        bool unresolved = false;
        for (auto tuple = bucket.rbegin(); tuple != bucket.rend(); ++tuple) {
            T color = tuple->color;
            U from = tuple->from;
            U to = tuple->to;
            if (from < to) std::swap(from, to);
            U area_root = this->canonize(area, from);

            if (to != area_root) {
                unresolved = true;
                area_bucket.push_back(Tuple<T, U>(color, area_root, color, from));
                area_bucket.push_back(Tuple<T, U>(color, area_root, color, to));
                area_bucket.push_back(Tuple<T, U>(color, from, color, area_root));
                area_bucket.push_back(Tuple<T, U>(color, to, color, area_root));
            } else {
                area_bucket.push_back(*tuple);
            }
        }

        return unresolved;
    };

    template <typename T, typename U=Parents::type>
    void resolve_roots(T color, TupleBuckets<T, U>& root_buckets, TupleBuckets<T, U>& area_buckets) {
        Tuples<T, U> tuples;
        Tuples<T, U>& root_tuples = root_buckets[color];
        Tuples<T, U>& area_tuples = area_buckets[color];

        tuples.reserve(root_tuples.size() / 2); // heuristic: about half the tuples are roots
        root_tuples.swap(tuples);
        area_tuples.clear();

        const size_t wrap_bound = std::numeric_limits<size_t>::max();
        const size_t tuple_count = tuples.size();

        U left_from = tuple_count > 0 ? tuples.front().from : std::numeric_limits<U>::max();
        U left_to = left_from;
        U right_from = tuple_count > 0 ? tuples.back().from : std::numeric_limits<U>::max();
        U right_to = right_from;

        // normalize tuples
        this->sample_sort(tuples);
        for (size_t i = 0; i < tuple_count; ++i) {
            auto& tuple = tuples[i];
            if (tuple.color != tuple.neighbor_color) continue;

            U from = tuple.from;
            U to = tuple.to;
            size_t j = i - 1;
            while (j != wrap_bound and tuples[j].from == from) {
                Tuple<T, U>& root_tuple = tuples[j];
                root_tuples.push_back(Tuple<T, U>(root_tuple.color, to, root_tuple.neighbor_color, root_tuple.to));
                --j;
            }
            if (j + 1 == 0) left_to = to;

            size_t k = i + 1;
            while (i + k < tuple_count and tuples[k].from == from) {
                Tuple<T, U>& inverse = tuples[k];
                T neighbor_color = inverse.neighbor_color;
                root_buckets[neighbor_color].push_back(Tuple<T, U>(neighbor_color, inverse.to, inverse.color, to));
                ++k;
            }
            if (k - 1 == tuple_count) right_to = to;

            i += k - 1;
            area_tuples.push_back(tuple);
        }

        // stitch the normalization at the bucket edges
        AreaEndpoints<U> ends = {
                AreaEndpoint<U>(left_from, left_to, left_to),
                AreaEndpoint<U>(right_from, right_to, right_to)
        };
        MPI_Scan(MPI_IN_PLACE, ends.data(), ends.size(), this->area_endpoint_type_, this->area_connect_, this->comm_);
        ends.front().to = ends.front().canonical_to;
        ends.back().to = ends.back().canonical_to;
        std::swap(ends.front(), ends.back());

        MPI_Scan(MPI_IN_PLACE, ends.data(), ends.size(), this->area_endpoint_type_, this->area_connect_, this->reverse_comm_);
        ends.front().to = ends.front().canonical_to;
        ends.back().to = ends.back().canonical_to;
        std::swap(ends.front(), ends.back());

        for (size_t i = 0; i < tuple_count and tuples[i].from == left_from; ++i) {
            Tuple<T, U>& tuple = tuples[i];
            if (tuple.color < tuple.neighbor_color) {
                T neighbor_color = tuple.neighbor_color;
                root_buckets[neighbor_color].push_back(Tuple<T, U>(neighbor_color, ends.front().to, tuple.color, tuple.from));
            } else if (tuple.color > tuple.neighbor_color) {
                root_tuples.push_back(Tuple<T, U>(tuple.color, ends.front().to, tuple.neighbor_color, tuple.to));
            }
        }

        for (size_t i = tuple_count - 1; i != wrap_bound and tuples[i].from == right_from; --i) {
            Tuple<T, U>& tuple = tuples[i];
            if (tuple.color < tuple.neighbor_color) {
                T neighbor_color = tuple.neighbor_color;
                root_buckets[neighbor_color].push_back(Tuple<T, U>(neighbor_color, ends.back().to, tuple.color, tuple.from));
            } else if (tuple.color > tuple.neighbor_color) {
                root_tuples.push_back(Tuple<T, U>(tuple.color, ends.back().to, tuple.neighbor_color, tuple.to));
            }
        }

        this->sample_sort(root_tuples);
        RootRules<T, U> roots;
        for (auto& tuple : root_tuples) {

        }

        if (color == 198) {
            std::cout << root_tuples << std::endl;
            std::cout << area_tuples << std::endl;
        }


        // sort all tuples globally and solve the root
//        this->sample_sort(tuples);
//
//        RootRules<T, U> roots;
//        for (auto& tuple : tuples) {
//
//        }
    };

//    template<typename T, typename U=Parents::type>
//    void resolve_partial_chain(Tuples<T, U>& tuples, AreaRules<U>& area, RootRules<T, U>& roots) {
//        // iterate over each tuple and find a root for it and link connected areas
//        for (auto& tuple : tuples) {
//            U from = tuple.from;
//            U to = tuple.to;
//            T color = tuple.color;
//            T neighbor_color = tuple.neighbor_color;
//
//            // skip inverse tuples
//            if (color < neighbor_color) {
//                continue;
//            }
//
//            // is this a tuple that joins two area of current color, remap them
//            if (color == neighbor_color) {
//                if (to > from) std::swap(from, to);
//                U canonical_point = this->canonize(area, from);
//
//                if (to == canonical_point) {
//                    area[from] = to;
//                } else {
//                    auto min_max = std::minmax(to, canonical_point);
//                    area[from] = min_max.first;
//                    area[min_max.second] = min_max.first;
//                }
//                continue;
//            }
//
//            U area_root = this->canonize(area, from);
//            // is there already a root for this tuple, if not just create one
//            auto tuple_root_it = roots.find(area_root);
//            if (tuple_root_it == roots.end()) {
//                roots[area_root] = Root<T, U>(neighbor_color, to);
//                continue;
//            }
//
//            // there is already a root, select either the closer one or join the two areas
//            Root<T, U>& root_tuple = tuple_root_it->second;
//            if (root_tuple.first < neighbor_color and neighbor_color < color) {
//                roots[area_root] = Root<T, U>(neighbor_color, to);
//                continue;
//            }
//
//            // correctly colored root we might need to create an area rule
//            U to_root = this->canonize(area, to);
//            U root_canonical_point = this->canonize(area, root_tuple.second);
//            if (root_tuple.first == neighbor_color and to_root != root_canonical_point) {
//                auto from_to = std::minmax(root_canonical_point, to_root);
//                area[from_to.second] = from_to.first;
//            }
//        }
//
//        if (this->size_ == 1) return;
//        // tuple chain is locally resolved, exchange information globally
//        Endpoints<T, U> ends;
//        this->initialize_ends(ends, tuples, area, roots);
//        MPI_Scan(MPI_IN_PLACE, ends.data(), ends.size(), this->endpoint_type_, this->level_connect_, this->comm_);
//        this->stitch_ends(tuples, ends, area, roots);
//
//        this->initialize_ends(ends, tuples, area, roots);
//        std::swap(ends.front(), ends.back());
//        MPI_Scan(MPI_IN_PLACE, ends.data(), ends.size(), this->endpoint_type_, this->level_connect_, this->reverse_comm_);
//        std::swap(ends.front(), ends.back());
//        this->stitch_ends(tuples, ends, area, roots);
//    }
//
//    template<typename T, typename U=Parents::type>
//    void initialize_ends(Endpoints<T, U>& ends, Tuples<T, U>& tuples, AreaRules<U>& area, RootRules<T, U>& roots) {
//        if (tuples.empty()) return;
//
//        Tuple<T, U>& front = tuples.front();
//        Endpoint<T, U>& left = ends.front();
//
//        left.color = front.color;
//        left.from = front.from;
//        left.start_root_color = front.neighbor_color;
//        left.start_root_to = front.to;
//
//        auto root_iter = roots.find(left.from);
//        Root<T, U>&& root = (root_iter != roots.end()) ? root_iter->second : Root<T, U>(front.color, front.from);
//        left.current_from = this->canonize(area, left.from);
//        left.current_root_color = root.first;
//        left.current_root_to = this->canonize(area, root.second);
//
//        Tuple<T, U>& back = tuples.back();
//        Endpoint<T, U>& right = ends.back();
//
//        right.color = back.color;
//        right.from = back.from;
//        right.start_root_color = back.neighbor_color;
//        right.start_root_to = back.to;
//
//        root_iter = roots.find(right.from);
//        root = (root_iter != roots.end()) ? root_iter->second : Root<T, U>(back.color, back.from);
//        right.current_from = this->canonize(area, right.from);
//        right.current_root_color = root.first;
//        right.current_root_to = this->canonize(area, root.second);
//    };
//
//    template<typename T, typename U=Parents::type>
//    void stitch_ends(Tuples<T, U>& tuples, Endpoints<T, U>& ends, AreaRules<U>& area, RootRules<T, U>& roots) {
//        if (tuples.empty()) return;
//
//        const Tuple<T, U>& front = tuples.front();
//        const Tuple<T, U>& back = tuples.back();
//
//        const Endpoint<T, U>& left = ends.front();
//        const Endpoint<T, U>& right = ends.back();
//
//        if (front.from != left.current_from) {
//            area[front.from] = left.current_from;
//        }
//        if (back.from != right.current_from) {
//            area[back.from] = right.current_from;
//        }
//
//        Root<T, U>& front_root = roots[front.from];
//        Root<T, U>& back_root = roots[back.from];
//
//        if (front_root.first == left.current_root_color and front_root.second > left.current_root_to) {
//            area[front_root.second] = left.current_root_to;
//        }
//        if (back_root.first == right.current_root_color and back_root.second > right.current_root_to) {
//            area[back_root.second] = right.current_root_to;
//        }
//
//        front_root.first = left.current_root_color;
//        front_root.second = left.current_root_to;
//        back_root.first = right.current_root_color;
//        back_root.second = right.current_root_to;
//    };
//
//    template<typename T, typename U=Parents::type>
//    bool remap_tuples(T color, TupleBuckets<T, U>& tuple_buckets, AreaRules<U>& area, RootRules<T, U>& roots) {
//        Tuples<T, U> bucket;
//        Tuples<T, U>& tuples = tuple_buckets[color];
//        bucket.swap(tuples);
//
//        bool unresolved = false;
//        for (auto tuple = bucket.rbegin(); tuple != bucket.rend(); ++tuple) {
//            U area_root = this->canonize(area, tuple->from);
//
//            // link tuple, normalize the pointed to target
//            if (tuple->color == tuple->neighbor_color) {
//                if (tuple->from > tuple->to) {
//                    if (tuple->to != area_root) {
//                        unresolved = true;
//                        tuple_buckets[color].push_back(Tuple<T, U>(color, area_root, color, tuple->from));
//                        tuple_buckets[color].push_back(Tuple<T, U>(color, area_root, color, tuple->to));
//                        tuple_buckets[color].push_back(Tuple<T, U>(color, tuple->from, color, area_root));
//                        tuple_buckets[color].push_back(Tuple<T, U>(color, tuple->to, color, area_root));
//                    } else {
//                        tuple_buckets[color].push_back(*tuple);
//                    }
//                } else {
//                    if (tuple->from != area_root) {
//                        unresolved = true;
//                        tuple_buckets[color].push_back(Tuple<T, U>(color, tuple->from, color, area_root));
//                        tuple_buckets[color].push_back(Tuple<T, U>(color, tuple->to, color, area_root));
//                        tuple_buckets[color].push_back(Tuple<T, U>(color, area_root, color, tuple->from));
//                        tuple_buckets[color].push_back(Tuple<T, U>(color, area_root, color, tuple->to));
//                    } else {
//                        tuple_buckets[color].push_back(*tuple);
//                    }
//                }
//                continue;
//            }
//
//            // a tuple that needs normalization, keep it in the current bucket and resolve further
//            if (tuple->from != area_root) {
//                tuple->from = area_root;
//                tuple_buckets[color].push_back(*tuple);
//                unresolved = true;
//                continue;
//            }
//
//            // we have an inverse tuple, put it back into its actual bucket
//            const auto root_iter = roots.find(tuple->from);
//            if (root_iter == roots.end()) {
//                Tuple<T, U> inverse(tuple->neighbor_color, tuple->to, color, area_root);
//                tuple_buckets[tuple->neighbor_color].push_back(inverse);
//                continue;
//            }
//
//            // a tuple with a target to high up in the tree compared  actual root, link the root and the target
//            const Root<T, U>& root = root_iter->second;
//            U to_root = this->canonize(area, tuple->to);
//
//            if (root.first > tuple->neighbor_color) {
//                Tuple<T, U> link_tuple(root.first, this->canonize(area, root.second), tuple->neighbor_color, to_root);
//                tuple_buckets[root.first].push_back(link_tuple);
//                if (tuple->to != to_root ) {
//                    T neighbor_color = tuple->neighbor_color;
//                    Tuples<T, U>& neighbor_bucket = tuple_buckets[neighbor_color];
//                    neighbor_bucket.push_back(Tuple<T, U>(neighbor_color, tuple->to, neighbor_color, to_root));
//                    neighbor_bucket.push_back(Tuple<T, U>(neighbor_color, to_root, neighbor_color, tuple->to));
//
//                }
//                continue;
//            }
//
//            // tuples with the correct colored roots, canonize it and push the inverse
//            tuple_buckets[color].push_back(Tuple<T, U>(color, tuple->from, tuple->neighbor_color, to_root));
//            if (tuple->to != to_root ) {
//                tuple_buckets[root.first].push_back(Tuple<T, U>(root.first, tuple->to, root.first, to_root));
//                tuple_buckets[root.first].push_back(Tuple<T, U>(root.first, to_root, root.first, tuple->to));
//            }
//        }
//
//        return unresolved;
//    };
//
//    template<typename T, typename U=Parents::type>
//    void final_remap(T color, TupleBuckets<T, U>& tuple_buckets, AreaRules<U>& area, RootRules<T, U>& roots) {
//        Tuples<T, U> bucket;
//        Tuples<T, U>& tuples = tuple_buckets[color];
//        bucket.swap(tuples);
//
//        for (auto tuple = bucket.rbegin(); tuple != bucket.rend(); ++tuple) {
//            U area_root = this->canonize(area, tuple->from);
//            U to_root = this->canonize(area, tuple->to);
//
//            // we have an inverse tuple, with an actual root
//            if (tuple->neighbor_color > tuple->color) {
//                Tuple<T, U> inverse(tuple->neighbor_color, tuple->to, color, area_root);
//                tuple_buckets[tuple->neighbor_color].push_back(inverse);
//                continue;
//            }
//
//            // an area tuple
//            if (tuple->neighbor_color == tuple->color and tuple->from < tuple->to) {
//                Tuple<T, U> inverse(tuple->neighbor_color, tuple->to, color, area_root);
//                tuple_buckets[tuple->neighbor_color].push_back(inverse);
//                continue;
//            }
//
//            const Root<T, U>& root = roots.find(tuple->from)->second;
//            // tuples with the correct colored roots, canonize it and push the inverse
//            if (tuple->to != to_root) {
//                tuple_buckets[root.first].push_back(Tuple<T, U>(root.first, tuple->to, root.first, to_root));
//            }
//
//            Tuples<T, U>& target_bucket = tuple_buckets[tuple->neighbor_color];
//            if (tuple->neighbor_color != color) {
//                target_bucket.push_back(Tuple<T, U>(tuple->neighbor_color, to_root, color, tuple->from));
//            } else {
//                tuple->to = to_root;
//                target_bucket.push_back(*tuple);
//            }
//        }
//    }
//
//    template<typename T, typename U=Parents::type>
//    void redistribute_tuples(const Image<T>& image, TupleBuckets<T, U>& tuple_buckets, Tuples<T, U>& incoming) {
//        if (this->size_ == 1) {
//            for (const auto& bucket : tuple_buckets) {
//                incoming.insert(incoming.end(), bucket.begin(), bucket.end());
//            }
//            return;
//        }
//
//        // determine the total number of pixels in the image
//        size_t total_height = image.height() - (this->rank_ + 1 != this->size_ ? 1 : 0);
//        MPI_Allreduce(MPI_IN_PLACE, &total_height, 1, MPI_Types<decltype(total_height)>::map(), MPI_SUM, this->comm_);
//
//        // chunk up the image to determine tuple target ranks
//        size_t width = image.width();
//        U total_tuples = 0;
//        U chunk = total_height / this->size_ * width;
//        U remainder = total_height % this->size_;
//
//        // calculate which tuple goes where
//        std::vector<int> send_counts(static_cast<size_t>(this->size_), 0);
//        std::vector<int> recv_counts(static_cast<size_t>(this->size_), 0);
//        for (const auto& bucket : tuple_buckets) {
//            for (const auto& tuple : bucket) {
//                U target_rank = tuple.from / chunk;
//                while (target_rank * chunk + std::min(target_rank, remainder) * width > tuple.from) {
//                    if (target_rank == 0) break;
//                    --target_rank;
//                }
//                ++send_counts[target_rank];
//                ++total_tuples;
//            }
//        }
//        // exchange the histogram with the other ranks
//        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, this->comm_);
//
//        //  calculate the buffer displacements
//        U total_received_tuples = recv_counts[0];
//        std::vector<int> send_displs(this->size_, 0);
//        std::vector<int> recv_displs(this->size_, 0);
//        for (size_t i = 1; i < send_displs.size(); ++i) {
//            total_received_tuples += recv_counts[i];
//            send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
//            recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
//        }
//
//        Tuples<T, U> outgoing(total_tuples);
//        incoming.resize(total_received_tuples);
//        std::vector<int> placement = send_displs;
//
//        for (auto& bucket : tuple_buckets) {
//            for (auto& tuple : bucket) {
//                U target_rank = tuple.from / chunk;
//                while (target_rank * chunk + std::min(target_rank, remainder) * width > tuple.from) {
//                    if (target_rank == 0) break;
//                    --target_rank;
//                }
//                int& position = placement[target_rank];
//                outgoing[position] = tuple;
//                ++position;
//            }
//            bucket.clear();
//        }
//
//        MPI_Alltoallv(outgoing.data(), send_counts.data(), send_displs.data(), this->tuple_type_,
//                      incoming.data(), recv_counts.data(), recv_displs.data(), this->tuple_type_, this->comm_);
//    };
//
//    template<typename T, typename U=Parents::type>
//    void generate_parent_image(Parents& parents, Tuples<T, U>& resolved, AreaRules<U>& area, size_t offset) {
//        RootRules<T, U> roots;
//        U nodes = 0;
//        area.clear();
//
//        // parse the tuples and store the results in the area/roots map
//        for (auto& tuple : resolved) {
//            if (tuple.color != tuple.neighbor_color) {
//                roots[tuple.from] = Root<T, U>(tuple.neighbor_color, tuple.to);
//                ++nodes;
//            } else {
//                area[tuple.from] = tuple.to;
//            }
//        }
//
//        // normalize the halo zone and communicate it to the neighbor nodes
//        size_t width = parents.width();
//        size_t total_pixels = width * parents.height();
//        size_t halo_offset = total_pixels - width;
//        std::vector<U> buffer(width);
//
//        for (size_t i = halo_offset; i < total_pixels; ++i) {
//            parents[i] = this->canonize(area, parents[i]);
//        }
//        MPI_Scan(parents.data() + halo_offset, buffer.data(), width, MPI_Types<U>::map(), this->right_stitch_, this->comm_);
//
//        // canonize the flat areas
//        for (auto& pixel : parents) {
//            pixel = this->canonize(area, pixel);
//        }
//
//        // and finally set the roots
//        for (const auto& root : roots) {
//            parents[root.first - offset] = root.second.second;
//        }
//
//        // compute the number of nodes
//        MPI_Reduce(this->rank_ == 0 ? MPI_IN_PLACE : &nodes, &nodes, 1, MPI_Types<U>::map(), MPI_SUM, 0, this->comm_);
//        if (this->rank_ == 0) {
//            ++nodes;
//            std::cout << "Number of nodes: " << nodes << std::endl;
//        }
//    };
};

#endif // DISTRIBUTED_MAX_TREE_H
