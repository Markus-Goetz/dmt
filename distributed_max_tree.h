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
    DistributedMaxTree(MPI_Comm comm = MPI_COMM_WORLD) : comm_(comm) {
        MPI_Comm_rank(comm, &this->rank_);
        MPI_Comm_size(comm, &this->size_);
    }

    template<typename T, typename U=Parents::type>
    Parents compute(const Image<T>& image) {
        AreaRules<U> area;

        // calculate the global pixel offset
        size_t offset = image.size() - image.width();
        MPI_Exscan(MPI_IN_PLACE, &offset, 1, MPI_Types<U>::map(), MPI_SUM, this->comm_);
        offset = this->rank_ == 0 ? 0 : offset;

        // allocate the parent image and buckets for the tuples
        Parents parents(Parents::infinity, image.width(), image.height());

        // MPI type and op creation
        MPI_Comm_split(this->comm_, 0, this->size_ - this->rank_ - 1, &this->reverse_comm_);
        Tuple<T, U>::create_mpi_type(&this->tuple_type_);
        AreaEndpoint<U>::create_mpi_type(&this->area_endpoint_type_);
        this->create_right_stitch_op<U>(parents, area);
        this->create_left_stitch_op<U>(parents, area);
        this->create_area_connect_op<U>();
        this->create_root_find_op<T, U>();

        // distributed resolution
        TupleBuckets<T, U> root_buckets = this->get_local_tuples(image, parents, offset);
        TupleBuckets<T, U> area_buckets = this->connect_halos(image, parents, offset, area, root_buckets.size());
        this->resolve_tuples(area_buckets, root_buckets);

        // send tuples to owners and apply
        Tuples<T, U> resolved_area;
        Tuples<T, U> resolved_roots;
        this->redistribute_tuples(image, area_buckets, resolved_area);
        this->redistribute_tuples(image, root_buckets, resolved_roots);
        this->generate_parent_image(parents, resolved_area, resolved_roots, offset);

        // MPI clean up
        MPI_Op_free(&this->area_connect_);
        MPI_Op_free(&this->root_find_);
        MPI_Op_free(&this->left_stitch_);
        MPI_Op_free(&this->right_stitch_);
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

    MPI_Op right_stitch_;
    MPI_Op left_stitch_;
    MPI_Op area_connect_;
    MPI_Op root_find_;

    int rank_;
    int size_;

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

            if (in_right.from == out_left.from or in_right.from == out_left.to) {
                out_left.canonical_to = std::min(in_right.canonical_to, out_left.canonical_to);
            }

            // very long chain? just copy the already merged values
            if (out_left.from == out_right.from) {
                out_right = out_left;
            }
        };

        MPI_Op_create(connect, false, &this->area_connect_);
    }

    template<typename T, typename U=Parents::type>
    void create_root_find_op() {
        auto connect = [](void* in_, void* out_, int*, MPI_Datatype*) {
            Endpoints<T, U>* in = reinterpret_cast<Endpoints<T, U>*>(in_);
            Endpoints<T, U>* out = reinterpret_cast<Endpoints<T, U>*>(out_);

            Tuple<T, U>& in_right = in->back();
            Tuple<T, U>& out_left = out->front();
            Tuple<T, U>& out_right = out->back();

            if (in_right.from == out_left.from and (
                   (in_right.neighbor_color > out_left.neighbor_color) or
                   (in_right.neighbor_color == out_left.neighbor_color and in_right.to < out_left.to)
                )) {
                out_left.neighbor_color = in_right.neighbor_color;
                out_left.to = in_right.to;
            }

            // very long chain? just copy the already merged values
            if (out_left.from == out_right.from) {
                out_right = out_left;
            }
        };

        MPI_Op_create(connect, false, &this->root_find_);
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

    template <typename T, typename U=Parents::type>
    U canonize(RootRules<T, U>& rules, U origin) {
        U destination = origin;
        auto it = rules.find(destination);

        while (it != rules.end() and it->second.second != destination) {
            destination = it->second.second;
            it = rules.find(destination);
        }
        return destination;
    };

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

            add_area_tuples_to_roots(root_bucket, area_bucket);
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
    void add_area_tuples_to_roots(Tuples<T, U>& root_bucket, Tuples<T, U>& area_bucket) {
        // put the correct area tuple into the root bucket
        for (const auto& tuple : area_bucket) {
            if (tuple.from > tuple.to) {
                root_bucket.push_back(tuple);
            }
        }
    }

    template <typename T, typename U=Parents::type>
    void resolve_roots(T color, TupleBuckets<T, U>& root_buckets, TupleBuckets<T, U>& area_buckets) {
        Tuples<T, U> tuples;
        Tuples<T, U>& root_tuples = root_buckets[color];
        Tuples<T, U>& area_tuples = area_buckets[color];

        tuples.reserve(root_tuples.size() / 2); // heuristic: about half the tuples are roots
        root_tuples.swap(tuples);
        area_tuples.clear();

        // normalize tuples
        this->sample_sort(tuples);
        const size_t wrap_bound = std::numeric_limits<size_t>::max();
        const size_t tuple_count = tuples.size();

        U left_from = tuple_count > 0 ? tuples.front().from : std::numeric_limits<U>::max();
        U left_to = left_from;
        U right_from = tuple_count > 0 ? tuples.back().from : std::numeric_limits<U>::max();
        U right_to = right_from;

        for (size_t i = 0; i < tuple_count; ++i) {
            Tuple<T, U>& tuple = tuples[i];
            if (tuple.from != left_from) break;
            if (tuple.color == tuple.neighbor_color) left_to = tuple.to;
        }
        for (size_t i = tuple_count - 1; i < wrap_bound; --i) {
            Tuple<T, U>& tuple = tuples[i];
            if (tuple.from != right_from) break;
            if (tuple.color == tuple.neighbor_color) right_to = tuple.to;
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

        for (size_t i = 0; i < tuple_count; ++i) {
            auto& tuple = tuples[i];
            U normalized = tuple.from;
            if (normalized == left_from) normalized = ends[0].canonical_to;
            else if (normalized == right_from) normalized = ends[1].canonical_to;

            // case 1: regular tuple, normalize from if at edge
            if (tuple.color > tuple.neighbor_color) {
                tuple.from = normalized;
                root_tuples.push_back(tuple);
            // case 2: area tuple, normalize left and right
            } else if (tuple.color == tuple.neighbor_color) {
                area_tuples.push_back(tuple);
                normalized = tuple.to;

                // correct the root tuples to the left
                size_t j = root_tuples.size() - 1;
                while (j != wrap_bound and root_tuples[j].from == tuple.from) {
                    Tuple<T, U>& root_tuple = root_tuples[j];
                    root_tuple.from = normalized;
                    --j;
                }

                // correct inverse tuples to the right
                size_t k = i + 1;
                while (k < tuple_count and tuples[k].from == tuple.from) {
                    Tuple<T, U>& inverse = tuples[k];
                    inverse.from = normalized;
                    std::swap(inverse.color, inverse.neighbor_color);
                    std::swap(inverse.from, inverse.to);
                    root_buckets[inverse.color].push_back(inverse);
                    ++k;
                }
                i = k - 1;
            // case 3: inverse tuple at edge, normalize from, invert and put pack in original bucket
            } else {
                tuple.from = normalized;
                std::swap(tuple.color, tuple.neighbor_color);
                std::swap(tuple.from, tuple.to);
                root_buckets[tuple.color].push_back(tuple);
            }
        }

        RootRules<T, U> roots;
        this->sample_sort(root_tuples);
        for (auto& tuple : root_tuples) {
            auto root_iter = roots.find(tuple.from);
            if (root_iter == roots.end()) {
                roots[tuple.from] = Root<T, U>(tuple.neighbor_color, tuple.to);
                continue;
            }

            Root<T, U>& root = root_iter->second;
            if (root.first < tuple.neighbor_color or (root.first == tuple.neighbor_color and root.second > tuple.to)) {
                root.first = tuple.neighbor_color;
                root.second = tuple.to;
            }
        }

        Endpoints<T, U> root_ends;
        this->initialize_ends(root_ends, root_tuples, roots);
        MPI_Scan(MPI_IN_PLACE, root_ends.data(), root_ends.size(), this->tuple_type_, this->root_find_, this->comm_);
        this->stitch_ends(root_ends, roots);

        this->initialize_ends(root_ends, root_tuples, roots);
        std::swap(root_ends.front(), root_ends.back());
        MPI_Scan(MPI_IN_PLACE, root_ends.data(), root_ends.size(), this->tuple_type_, this->root_find_, this->reverse_comm_);
        std::swap(root_ends.front(), root_ends.back());
        this->stitch_ends(root_ends, roots);

        this->remap_root_tuples(color, root_buckets, area_buckets, roots);
    };

    template<typename T, typename U=Parents::type>
    void initialize_ends(Endpoints<T, U>& ends, Tuples<T, U>& tuples, RootRules<T, U>& roots) {
        if (tuples.empty()) return;

        Tuple<T, U>& front = tuples.front();
        Tuple<T, U>& left = ends.front();
        left = front;

        Root<T, U>& left_root = roots.find(left.from)->second;
        left.neighbor_color = left_root.first;
        left.to = left_root.second;

        Tuple<T, U>& back = tuples.back();
        Tuple<T, U>& right = ends.back();
        right = back;

        Root<T, U>& right_root = roots.find(right.from)->second;
        right.neighbor_color = right_root.first;
        right.to = right_root.second;
    };

    template <typename T, typename U=Parents::type>
    void stitch_ends(Endpoints<T, U>& ends, RootRules<T, U>& roots) {
        const Tuple<T, U>& left = ends.front();
        roots[left.from] = Root<T, U>(left.neighbor_color, left.to);

        const Tuple<T, U>& right = ends.back();
        roots[right.from] = Root<T, U>(right.neighbor_color, right.to);
    };

    template <typename T, typename U=Parents::type>
    void remap_root_tuples(T color, TupleBuckets<T, U>& root_buckets, TupleBuckets<T, U>& area_buckets, RootRules<T, U>& roots) {
        RootRules<T, U> area;
        Tuples<T, U> remapped;
        Tuples<T, U>& tuples = root_buckets[color];
        remapped.swap(tuples);

        Root<T, U> root;
        U from = std::numeric_limits<U>::max();
        for (Tuple<T, U>& tuple : remapped) {
            if (tuple.from != from) {
                from = tuple.from;
                root = roots.find(from)->second;
            }

            // case 1: inverse tuple, put it back to its original bucket
            if (tuple.neighbor_color > tuple.color) {
                root_buckets[tuple.neighbor_color].push_back(Tuple<T, U>(tuple.neighbor_color, tuple.to, tuple.color, tuple.from));
            // case 2: root to high up in the chain
            } else if (tuple.neighbor_color < root.first) {
                root_buckets[root.first].push_back(Tuple<T, U>(root.first, root.second, tuple.neighbor_color, tuple.to));
            // case 3: color of root is correct, but canonical point does not fit, create area tuple
            } else if (tuple.neighbor_color == root.first and root.second < tuple.to) {
                U canonized_to = this->canonize(area, tuple.to);
                U canonized_root = this->canonize(area, root.second);
                auto area_choice = std::minmax(canonized_to, canonized_root);
                // transitive closure
                area[tuple.to] = Root<T, U>(tuple.neighbor_color, area_choice.first);
                area[area_choice.second] = Root<T, U>(tuple.neighbor_color, area_choice.first);
            // case 4: invert correct tuple for normalization
            } else {
                root_buckets[root.first].push_back(Tuple<T, U>(root.first, root.second, tuple.color, tuple.from));
            }
        }

        for (auto& rule : area) {
            T tuple_color = rule.second.first;
            U from = rule.first;
            U to = rule.second.second;

            Tuples<T, U>& area_bucket = area_buckets[tuple_color];
            area_bucket.push_back(Tuple<T, U>(tuple_color, from, tuple_color, to));
            area_bucket.push_back(Tuple<T, U>(tuple_color, to, tuple_color, from));
        }
    };

    template<typename T, typename U=Parents::type>
    void redistribute_tuples(const Image<T>& image, TupleBuckets<T, U>& buckets, Tuples<T, U>& resolved) {
        if (this->size_ == 1) {
            for (auto& bucket : buckets) {
                resolved.insert(resolved.end(), bucket.begin(), bucket.end());
            }
            return;
        }

        // determine the total number of pixels in the image
        size_t total_height = image.height() - (this->rank_ + 1 != this->size_ ? 1 : 0);
        MPI_Allreduce(MPI_IN_PLACE, &total_height, 1, MPI_Types<decltype(total_height)>::map(), MPI_SUM, this->comm_);

        // chunk up the image to determine tuple target ranks
        size_t width = image.width();
        U total_tuples = 0;
        U chunk = total_height / this->size_ * width;
        U remainder = total_height % this->size_;

        // calculate which tuple goes where
        std::vector<int> send_counts(static_cast<size_t>(this->size_), 0);
        std::vector<int> recv_counts(static_cast<size_t>(this->size_), 0);
        for (const auto& bucket : buckets) {
            for (const auto& tuple : bucket) {
                U target_rank = tuple.from / chunk;
                while (target_rank * chunk + std::min(target_rank, remainder) * width > tuple.from) {
                    if (target_rank == 0) break;
                    --target_rank;
                }
                ++send_counts[target_rank];
                ++total_tuples;
            }
        }
        // exchange the histogram with the other ranks
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, this->comm_);

        //  calculate the buffer displacements
        U total_received_tuples = recv_counts[0];
        std::vector<int> send_displs(this->size_, 0);
        std::vector<int> recv_displs(this->size_, 0);
        for (size_t i = 1; i < send_displs.size(); ++i) {
            total_received_tuples += recv_counts[i];
            send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
            recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
        }

        Tuples<T, U> outgoing(total_tuples);
        resolved.resize(total_received_tuples);
        std::vector<int> placement = send_displs;

        for (auto& bucket : buckets) {
            for (auto& tuple : bucket) {
                U target_rank = tuple.from / chunk;
                while (target_rank * chunk + std::min(target_rank, remainder) * width > tuple.from) {
                    if (target_rank == 0) break;
                    --target_rank;
                }
                int& position = placement[target_rank];
                outgoing[position] = tuple;
                ++position;
            }
            bucket.clear();
        }

        MPI_Alltoallv(outgoing.data(), send_counts.data(), send_displs.data(), this->tuple_type_,
                      resolved.data(), recv_counts.data(), recv_displs.data(), this->tuple_type_, this->comm_);
    };

    template<typename T, typename U=Parents::type>
    void generate_parent_image(Parents& parents, Tuples<T, U>& resolved_area, Tuples<T, U>& resolved_roots, size_t offset) {
        // parse the area tuples and memorize area rules
        AreaRules<U> area;
        for (const auto& tuple : resolved_area) {
            area[tuple.from] = tuple.to;
        }

        // normalize the halo zone and communicate it to the neighbor nodes
        size_t width = parents.width();
        size_t total_pixels = width * parents.height();
        size_t halo_offset = total_pixels - width;
        std::vector<U> buffer(width);

        for (size_t i = halo_offset; i < total_pixels; ++i) {
            parents[i] = this->canonize(area, parents[i]);
        }
        MPI_Scan(parents.data() + halo_offset, buffer.data(), width, MPI_Types<U>::map(), this->right_stitch_, this->comm_);

        // canonize the flat areas
        for (auto& pixel : parents) {
            pixel = this->canonize(area, pixel);
        }

        // and finally set the roots
        for (const auto& root : resolved_roots) {
            parents[root.from - offset] = root.to;
        }

        // compute the number of nodes
        U nodes = resolved_roots.size();
        MPI_Reduce(this->rank_ == 0 ? MPI_IN_PLACE : &nodes, &nodes, 1, MPI_Types<U>::map(), MPI_SUM, 0, this->comm_);
        if (this->rank_ == 0) {
            ++nodes;
            std::cout << "Number of nodes: " << nodes << std::endl;
        }
    };
};

#endif // DISTRIBUTED_MAX_TREE_H