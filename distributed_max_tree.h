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
#include <vector>
#include <unordered_set>

#include <mpi.h>

#include "endpoint.h"
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
using AreaRules = std::map<T, T>;

template <typename T, typename U=Parents::type>
using Root = std::pair<T, U>;

template <typename T, typename U=Parents::type>
using RootRules = std::map<U, Root<T, U>>;

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
        AreaRules<U> area_rules;

        // create the reverse communicator used for the stitching scans
        MPI_Comm_split(this->comm_, 0, this->size_ - this->rank_ - 1, &this->reverse_comm_);

        // calculate the global pixel offset
        size_t offset = image.size() - image.width();
        MPI_Exscan(MPI_IN_PLACE, &offset, 1, MPI_Types<U>::map(), MPI_SUM, this->comm_);
        offset = this->rank_ == 0 ? 0 : offset;

        // allocate the parent image and buckets for the tuples
        Parents parents(Parents::infinity, image.width(), image.height());
        // TupleBuckets<T, U> tuple_buckets(Image<T>::infinity);
        TupleBuckets<T, U> tuple_buckets(Image<T>::infinity + 1); // GC

        // MPI type and op creation
        Tuple<T, U>::create_mpi_type(&this->tuple_type_);
        Endpoint<T, U>::create_mpi_type(&this->endpoint_type_);
        this->create_right_stitch_op<U>(parents, area_rules);
        this->create_left_stitch_op<U>(parents, area_rules);
        this->create_level_connect_op<T, U>();

        // here is the meat!
        Tuples<T, U> resolved_tuples;
        this->get_local_tuples(image, parents, tuple_buckets, offset);
        std::cout << 1 << std::endl;
        this->connect_halos(parents, area_rules);
        std::cout << 2 << std::endl;
        this->resolve_tuples(tuple_buckets, area_rules);
        std::cout << 3 << std::endl;
        this->redistribute_tuples(image, tuple_buckets, resolved_tuples);
        std::cout << 4 << std::endl;
        this->generate_parent_image(parents, resolved_tuples, area_rules, offset);
        std::cout << 5 << std::endl;

        // MPI clean up
        MPI_Op_free(&this->level_connect_);
        MPI_Op_free(&this->left_stitch_);
        MPI_Op_free(&this->right_stitch_);
        MPI_Type_free(&this->endpoint_type_);
        MPI_Type_free(&this->tuple_type_);
        MPI_Comm_free(&this->reverse_comm_);

        return parents;
    }

protected:
    MPI_Comm comm_;
    MPI_Comm reverse_comm_;
    MPI_Datatype tuple_type_;
    MPI_Datatype endpoint_type_;
    MPI_Op right_stitch_;
    MPI_Op left_stitch_;
    MPI_Op level_connect_;

    int rank_;
    int size_;

    template<typename U=Parents::type>
    void create_right_stitch_op(const Parents& parents, AreaRules<U>& rules) {
        auto right_rule_scan = [this, &parents, &rules](void* in_, void* out_, int* len, MPI_Datatype* type) {
            U* in = reinterpret_cast<U*>(in_);
            U* out = reinterpret_cast<U*>(out_);

            for (int i = 0; i < *len; ++i) {
                U remote = in[i];
                U local = parents[i];
                auto it = rules.find(local);

                if (it != rules.end() and it->second != remote) {
                    rules[local] = std::min(
                            this->canonize<U>(rules, remote),
                            this->canonize<U>(rules, it->second)
                    );
                } else if (local != remote) {
                    rules[local] = remote;
                }
            }

            size_t offset = parents.width() * parents.height() - parents.width();
            for (int i = 0; i < *len; ++i) {
                out[i] = this->canonize<U>(rules, parents[offset + i]);
            }
        };
        static auto static_right_scan = right_rule_scan;

        MPI_Op_create([](void* in, void* out, int* len, MPI_Datatype* type) {
            static_right_scan(in, out, len, type);
        }, false, &this->right_stitch_);
    }

    template<typename U=Parents::type>
    void create_left_stitch_op(const Parents& parents, AreaRules<U>& rules) {
        auto left_rule_scan = [this, &parents, &rules](void* in_, void* out_, int* len, MPI_Datatype* type) {
            U* in = reinterpret_cast<U*>(in_);
            U* out = reinterpret_cast<U*>(out_);
            size_t offset = parents.width() * parents.height() - parents.width();

            for (int i = 0; i < *len; ++i) {
                U remote = in[i];
                U local = parents[offset + i];

                if (local != remote) {
                    rules[local] = remote;
                }
            }

            for (int i = 0; i < *len; ++i) {
                out[i] = this->canonize<U>(rules, parents[i]);
            }
        };
        static auto static_left_scan = left_rule_scan;

        MPI_Op_create([](void* in, void* out, int* len, MPI_Datatype* type) {
            static_left_scan(in, out, len, type);
        }, false, &this->left_stitch_);
    }

    template<typename T, typename U=Parents::type>
    void create_level_connect_op() {
        auto connect = [](void* in_, void* out_, int* len, MPI_Datatype* type) {
            Endpoint<T, U>* in = reinterpret_cast<Endpoint<T, U>*>(in_);
            Endpoint<T, U>* out = reinterpret_cast<Endpoint<T, U>*>(out_);

            Endpoint<T, U>& out_left = out[0];
            Endpoint<T, U>& out_right = out[1];
            Endpoint<T, U>& in_right = in[1];

            if (in_right.from == out_left.from) {
                out_left.current_from = std::min(in_right.current_from, out_left.current_from);

                if ((out_left.from == out_left.current_root_to) or
                    (in_right.current_root_color > out_left.current_root_color) or
                    (in_right.current_root_color == out_left.current_root_color and
                     in_right.current_root_to < out_left.current_root_to)) {
                    out_left.current_root_color = in_right.current_root_color;
                    out_left.current_root_to = in_right.current_root_to;
                }
            } else if (in_right.start_root_to == out_left.current_root_to and
                       in_right.current_root_color == in_right.start_root_color) {
                out_left.current_root_to = in_right.current_root_to;
            }

            // very long chain? just copy the already merged values
            if (out_left.from == out_right.from) {
                out_right = out_left;
            }
        };

        MPI_Op_create(connect, false, &this->level_connect_);
    }

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
    void get_local_tuples(const Image<T>& image, Parents& parents, TupleBuckets<T, U>& tuple_buckets, size_t offset) {
        // bucket sort image
        Buckets<U> buckets(Image<T>::infinity + 1); // Added by GC
        for (size_t i = 0; i < image.size(); ++i) {
            T color = image[i];
            buckets[color].push_back(i);
        }

        std::map<U, std::set<U> > outbound;
        T color = static_cast<T>(buckets.size() - 1);
        // connected component labeling
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
                        Tuples<T, U>& bucket = tuple_buckets[neighbor_color];
                        bucket.push_back(Tuple<T, U>(neighbor_color, neighbor_root, color, root));
                    }
                }
            }

            // advance to the next color channel
            --color;

        }
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

    template<typename U=Parents::type>
    void connect_halos(Parents& parents, AreaRules<U>& rules) {
        size_t width = parents.width();
        std::vector<U> send_buffer(width);
        std::vector<U> recv_buffer(width);
        size_t offset = width * parents.height() - width;

        // right directed communication of minimum
        U* read = parents.data() + (this->rank_ != 0 ? 0 : offset);
        MPI_Scan(read, recv_buffer.data(), width,
                 MPI_Types<U>::map(), this->right_stitch_, this->comm_);

        // left directed, backwards propagation of right minimum scan results
        for (size_t i = 0; i < width; ++i) {
            send_buffer[i] = this->canonize(rules, parents[i + (this->rank_ + 1 == this->size_ ? 0 : offset)]);
        }
        MPI_Scan(send_buffer.data(), recv_buffer.data(), width,
                 MPI_Types<U>::map(), this->left_stitch_, this->reverse_comm_);
    }

    template<typename T, typename U=Parents::type>
    void resolve_tuples(TupleBuckets<T, U>& tuple_buckets, AreaRules<U>& halo_area) {
        for (size_t i = Image<T>::infinity; i != std::numeric_limits<size_t>::max(); --i) { // Added by GC
            // retrieve the current bucket
            T color = static_cast<T>(i);
            Tuples<T, U>& bucket = tuple_buckets[color];

            // check whether there are tuples in this channel at all, if none are available, skip the channel
            size_t total = bucket.size();
            MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_UNSIGNED_LONG, MPI_SUM, this->comm_);
            if (total == 0) {
                continue;
            }

            // normalize the tuples
            if (!halo_area.empty()) {
                for (auto& tuple : bucket) {
                    tuple.from = this->canonize(halo_area, tuple.from);
                    tuple.to = this->canonize(halo_area, tuple.to);
                }
            }

            // resolve the tuple chains
            AreaRules<U> area;
            RootRules<T, U> roots;
            bool unresolved = true;
            while (unresolved) {
                area.clear(); roots.clear();
                this->sample_sort(bucket);
                this->resolve_partial_chain(color, bucket, tuple_buckets, area, roots);
                unresolved = this->remap_tuples(color, tuple_buckets, area, roots);
                // globally done?
                MPI_Allreduce(MPI_IN_PLACE, &unresolved, 1, MPI_C_BOOL, MPI_LOR, this->comm_);
            }
            this->remap_tuples_last(color, tuple_buckets, area, roots);
        }
    }

    template <typename T, typename U=Parents::type>
    void balance_tuples(Tuples<T, U>& bucket) {
        size_t local_elements = bucket.size();
        size_t total_elements = local_elements;
        size_t left_elements = local_elements;

        MPI_Allreduce(MPI_IN_PLACE, &total_elements, 1, MPI_Types<size_t>::map(), MPI_SUM, this->comm_);
        MPI_Exscan(MPI_IN_PLACE, &left_elements, 1, MPI_Types<size_t>::map(), MPI_SUM, this->comm_);
        left_elements = this->rank_ != 0 ? left_elements : 0;

        size_t chunk = total_elements / this->size_;
        size_t remainder = total_elements % this->size_;

        int target_rank = static_cast<int>(left_elements / chunk);
        if (target_rank * chunk + std::min(static_cast<size_t>(target_rank), remainder) > left_elements) {
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

        Tuples<T, U> incoming(chunk + (this->rank_ < remainder ? 1 : 0));
        MPI_Alltoallv(bucket.data(), send_counts.data(), send_displs.data(), this->tuple_type_,
                      incoming.data(), recv_counts.data(), recv_displs.data(), this->tuple_type_, this->comm_);
        bucket.swap(incoming);
    };

    template<typename T, typename U=Parents::type>
    void sample_sort(Tuples<T, U>& bucket) {
        this->balance_tuples(bucket);
        // allocate space for the splitters and prepare a no-op dummy
        const size_t split_count = this->size_ - 1;
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
        MPI_Allgather(MPI_IN_PLACE, split_count, this->tuple_type_,
                      splitters.data(), split_count, this->tuple_type_, this->comm_);

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
            while (tuple >= current_split) {
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
        for (int i = 1; i < this->size_ - 1; ++i) {
            std::inplace_merge(incoming_tuples.begin(), incoming_tuples.begin() + recv_displs[i], incoming_tuples.begin() + recv_displs[i + 1]);
        }
        incoming_tuples.erase(std::unique(incoming_tuples.begin(), incoming_tuples.end()), incoming_tuples.end());

        // exchange the incoming tuples with the bucket
        bucket.swap(incoming_tuples);
        this->balance_tuples(bucket);
    };

    template<typename T, typename U=Parents::type>
    void resolve_partial_chain(T color, Tuples<T, U>& tuples, TupleBuckets<T, U>& tuple_buckets,
                               AreaRules<U>& area, RootRules<T, U>& roots) {
        // iterate over each tuple and find a root for it and link connected areas
        for (auto& tuple : tuples) {
            const U from = tuple.from;
            const U to = tuple.to;
            const T color = tuple.color;
            const T neighbor_color = tuple.neighbor_color;
            const U area_root = this->canonize(area, from);

            // neighbor color is larger, skip over it while marking roots, check whether it is not normalized though
            if (color < neighbor_color) {
                continue;
            }

            // is this a tuple that joins two area of current color, remap them
            if (color == neighbor_color) {
                area[from] = this->canonize(area, to);
            }

            // is there already a root for this tuple, if not just create one
            auto tuple_root_it = roots.find(area_root);
            if (tuple_root_it == roots.end()) {
                roots[area_root] = Root<T, U>(neighbor_color, to);
                continue;
            }

            // there is already a root, select either the closer one or join the two areas
            Root<T, U>& root_tuple = tuple_root_it->second;
            if (root_tuple.first < neighbor_color) {
                roots[area_root] = Root<T, U>(neighbor_color, to);
                continue;
            }

            U to_root = this->canonize(area, to);
            U root_root = this->canonize(area, root_tuple.second);
            if (root_tuple.first == neighbor_color and to_root != root_root) {
                auto from_to = std::minmax(root_root, to_root);
                area[from_to.second] = from_to.first;
            }
        }

        // tuple chain is locally resolved, exchange information globally
        Endpoints<T, U> ends;
        this->initialize_ends(ends, tuples, area, roots);
        MPI_Scan(MPI_IN_PLACE, ends.data(), ends.size(), this->endpoint_type_, this->level_connect_, this->comm_);
        this->stitch_ends(tuples, ends, area, roots);

        this->initialize_ends(ends, tuples, area, roots);
        std::swap(ends.front(), ends.back());
        MPI_Scan(MPI_IN_PLACE, ends.data(), ends.size(), this->endpoint_type_, this->level_connect_, this->reverse_comm_);
        std::swap(ends.front(), ends.back());
        this->stitch_ends(tuples, ends, area, roots);
    }

    template<typename T, typename U=Parents::type>
    void initialize_ends(Endpoints<T, U>& ends, Tuples<T, U>& tuples, AreaRules<U>& area, RootRules<T, U>& roots) {
        if (tuples.empty()) return;

        Tuple<T, U>& front = tuples.front();
        Endpoint<T, U>& left = ends.front();

        left.color = front.color;
        left.from = front.from;
        left.start_root_color = front.neighbor_color;
        left.start_root_to = front.to;

        auto root_iter = roots.find(left.from);
        Root<T, U>&& root = (root_iter != roots.end()) ? root_iter->second : Root<T, U>(front.color, front.from);
        left.current_from = this->canonize(area, left.from);
        left.current_root_color = root.first;
        left.current_root_to = this->canonize(area, root.second);

        Tuple<T, U>& back = tuples.back();
        Endpoint<T, U>& right = ends.back();

        right.color = back.color;
        right.from = back.from;
        right.start_root_color = back.neighbor_color;
        right.start_root_to = back.to;

        root_iter = roots.find(right.from);
        root = (root_iter != roots.end()) ? root_iter->second : Root<T, U>(back.color, back.from);
        right.current_from = this->canonize(area, right.from);
        right.current_root_color = root.first;
        right.current_root_to = this->canonize(area, root.second);
    };

    template<typename T, typename U=Parents::type>
    void stitch_ends(Tuples<T, U>& tuples, Endpoints<T, U>& ends, AreaRules<U>& area, RootRules<T, U>& roots) {
        if (tuples.empty()) return;

        const Tuple<T, U>& front = tuples.front();
        const Tuple<T, U>& back = tuples.back();

        const Endpoint<T, U>& left = ends.front();
        const Endpoint<T, U>& right = ends.back();

        if (front.from != left.current_from) {
            area[front.from] = left.current_from;
        }
        if (back.from != right.current_from) {
            area[back.from] = right.current_from;
        }

        Root<T, U>& front_root = roots[front.from];
        Root<T, U>& back_root = roots[back.from];

        if (front_root.first == left.current_root_color and front_root.second > left.current_root_to) {
            area[front_root.second] = left.current_root_to;
        }
        if (back_root.first == right.current_root_color and back_root.second > right.current_root_to) {
            area[back_root.second] = right.current_root_to;
        }

        front_root.first = left.current_root_color;
        front_root.second = left.current_root_to;
        back_root.first = right.current_root_color;
        back_root.second = right.current_root_to;
    };


    template<typename T, typename U=Parents::type>
    void remap_tuples_last(T color, TupleBuckets<T, U>& tuple_buckets, AreaRules<U>& area, RootRules<T, U>& roots) {
        std::unordered_set<U> keep;
        std::unordered_set<U> linked;

        Tuples<T, U> bucket;
        Tuples<T, U>& tuples = tuple_buckets[color];
        bucket.swap(tuples);

        for (auto tuple = bucket.rbegin(); tuple != bucket.rend(); ++tuple) {
            U area_root = this->canonize(area, tuple->from);

            // we have an inverse tuple, put it back into its actual bucket
            const auto root_iter = roots.find(tuple->from);
            if (root_iter == roots.end()) {
                Tuple<T, U> inverse(tuple->neighbor_color, tuple->to, color, area_root);
                tuple_buckets[tuple->neighbor_color].push_back(inverse);
                continue;
            }

            const Root<T, U>& root = root_iter->second;
            U to_root = this->canonize(area, tuple->to);

            // tuples with the correct colored roots, canonize it and  push the inverse
            if (tuple->to != to_root and linked.find(tuple->to) == linked.end()) {
                linked.insert(tuple->to);
                tuple_buckets[root.first].push_back(Tuple<T, U>(root.first, tuple->to, root.first, to_root));
            }

            if (tuple->neighbor_color != color) {
                tuple_buckets[tuple->neighbor_color].push_back(
                        Tuple<T, U>(tuple->neighbor_color, to_root, color, tuple->from));
            } else {
                tuple_buckets[tuple->neighbor_color].push_back(
                        Tuple<T, U>(color, tuple->from, tuple->neighbor_color, to_root));
            }
        }
    }


    template<typename T, typename U=Parents::type>
    bool remap_tuples(T color, TupleBuckets<T, U>& tuple_buckets, AreaRules<U>& area, RootRules<T, U>& roots) {
        bool unresolved = false;
        std::unordered_set<U> keep;
        std::unordered_set<U> linked;

        Tuples<T, U> bucket;
        Tuples<T, U>& tuples = tuple_buckets[color];
        bucket.swap(tuples);

        for (auto tuple = bucket.rbegin(); tuple != bucket.rend(); ++tuple) {
            U area_root = this->canonize(area, tuple->from);

            // link tuple, normalize the pointed to target
            if (tuple->color == tuple->neighbor_color) {
                tuple->to = this->canonize(area, tuple->to);
                tuple_buckets[color].push_back(*tuple);
                continue;
            }

            // a tuple that needs normalization, keep it in the current bucket
            if (tuple->from != area_root or keep.find(area_root) != keep.end()) {
                keep.insert(area_root);
                tuple->from = area_root;
                tuples.push_back(*tuple);
                unresolved = true;
                continue;
            }

            // we have an inverse tuple, put it back into its actual bucket
            const auto root_iter = roots.find(tuple->from);
            if (root_iter == roots.end()) {
                Tuple<T, U> inverse(tuple->neighbor_color, tuple->to, color, area_root);
                tuple_buckets[tuple->neighbor_color].push_back(inverse);
                continue;
            }

            // a tuple with a target to high up in the tree compared  actual root, link the root and the target
            const Root<T, U>& root = root_iter->second;
            U to_root = this->canonize(area, tuple->to);

            if (root.first > tuple->neighbor_color) {
                Tuple<T, U> link_tuple(root.first, this->canonize(area, root.second), tuple->neighbor_color, to_root);
                tuple_buckets[root.first].push_back(link_tuple);
                if (tuple->to != to_root and linked.find(tuple->to) == linked.end()) {
                    linked.insert(tuple->to);

                    tuple_buckets[tuple->neighbor_color].push_back(
                            Tuple<T, U>(tuple->neighbor_color, tuple->to, tuple->neighbor_color, to_root));

                }
                continue;
            }

            // tuples with the correct colored roots, canonize it and  push the inverse
            if (tuple->to != to_root and linked.find(tuple->to) == linked.end()) {
                linked.insert(tuple->to);
                tuple_buckets[root.first].push_back(Tuple<T, U>(root.first, tuple->to, root.first, to_root));
            }

            // tuple_buckets[tuple->neighbor_color].push_back(Tuple<T, U>(tuple->neighbor_color, to_root, color, tuple->from));
            tuple_buckets[color].push_back(
                    Tuple<T, U>(color, tuple->from, tuple->neighbor_color, to_root));  // Added by GC

        }

        return unresolved;
    };

    template<typename T, typename U=Parents::type>
    void redistribute_tuples(const Image<T>& image, TupleBuckets<T, U>& tuple_buckets, Tuples<T, U>& incoming) {
        // determine the total number of pixels in the image
        size_t total_height = image.height() - (this->rank_ + 1 != this->size_ ? 1 : 0);
        MPI_Allreduce(MPI_IN_PLACE, &total_height, 1, MPI_UNSIGNED_LONG, MPI_SUM, this->comm_);

        // chunk up the image to determine tuple target ranks
        U total_tuples = 0;
        U chunk = total_height / this->size_ * image.width();
        U remainder = total_height % this->size_ * image.width();

        // calculate which tuple goes where
        std::vector<int> send_counts(this->size_, 0);
        std::vector<int> recv_counts(this->size_);
        for (const auto& bucket : tuple_buckets) {
            for (const auto& tuple : bucket) {
                U target_rank = tuple.from / chunk;
                if (target_rank * chunk + target_rank * remainder > tuple.from) {
                    --target_rank;
                }
                ++send_counts[target_rank];
                ++total_tuples;
            }
        }
        // exchange the histogram with the other ranks
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, this->comm_);

        // calculate the buffer displacements
        U total_received_tuples = recv_counts[0];
        std::vector<int> send_displs(this->size_, 0);
        std::vector<int> recv_displs(this->size_, 0);
        for (size_t i = 1; i < send_displs.size(); ++i) {
            total_received_tuples += recv_counts[i];
            send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
            recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
        }

        Tuples<T, U> outgoing(total_tuples);
        incoming.resize(total_received_tuples);
        std::vector<int> placement = send_displs;

        for (auto& bucket : tuple_buckets) {
            for (auto& tuple : bucket) {
                U target_rank = tuple.from / chunk;
                if (target_rank * chunk + target_rank * remainder > tuple.from) {
                    --target_rank;
                }
                int& position = placement[target_rank];
                outgoing[position] = tuple;
                ++position;
            }
            bucket.clear();
        }

        MPI_Alltoallv(outgoing.data(), send_counts.data(), send_displs.data(), this->tuple_type_,
                      incoming.data(), recv_counts.data(), recv_displs.data(), this->tuple_type_, this->comm_);
    };

    template<typename T, typename U=Parents::type>
    void generate_parent_image(Parents& parents, Tuples<T, U>& resolved, AreaRules<U>& area, size_t offset) {
        RootRules<T, U> roots;
        int local_nodes = 0, global_nodes = 0; // Added by GC

        // parse the tuples and store the results in the area/roots map
        for (auto& tuple : resolved) {
            if (tuple.color != tuple.neighbor_color) {
                roots[tuple.from] = Root<T, U>(tuple.neighbor_color, tuple.to);
                local_nodes++; // Added by GC
            } else {
                area[tuple.from] = tuple.to;
            }
        }

        // Compute the number of nodes  // Added by GC
        MPI_Reduce(&local_nodes, &global_nodes, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (this->rank_ == 0)
            std::cout << "Number of nodes: " << global_nodes + 1 << std::endl;

        // normalize the halo zone and communicate it to the neighbor nodes
        size_t width = parents.width();
        size_t total_pixels = width * parents.height();
        size_t halo_offset = total_pixels - width;

        for (size_t i = halo_offset; i < total_pixels; ++i) {
            parents[i] = this->canonize(area, parents[i]);
        }

        // exchange globally
        std::vector<U> recv_buffer(width);
        U* read = parents.data() + (this->rank_ != 0 ? 0 : halo_offset);
        MPI_Scan(read, recv_buffer.data(), width, MPI_Types<U>::map(), this->right_stitch_, this->comm_);

        // canonize the flat areas
        for (auto& pixel : parents) {
            pixel = this->canonize(area, pixel);
        }

        // and finally set the roots
        for (const auto& root : roots) {
            parents[root.first - offset] = root.second.second;
        }
    };
};

#endif // DISTRIBUTED_MAX_TREE_H{
