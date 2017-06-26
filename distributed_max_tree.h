#ifndef DISTRIBUTED_MAX_TREE_H
#define DISTRIBUTED_MAX_TREE_H

#include <cmath>
#include <cstdint>
#include <map>
#include <iostream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <mpi.h>

#include "endpoints.h"
#include "image.h"
#include "max_tree.h"
#include "mpi_wrapper.h"
#include "rules.h"
#include "tuple.h"
#include "util.h"

class DistributedMaxTree {
public:
    DistributedMaxTree(MPI_Comm comm = MPI_COMM_WORLD, unsigned int thread_count=std::thread::hardware_concurrency())
            : comm_(comm), id_(std::this_thread::get_id()), thread_count_(thread_count) {
        MPI_Comm_rank(comm, &this->rank_);
        MPI_Comm_size(comm, &this->size_);
    }

    template<typename T, typename U=Parents::type>
    Parents compute(const Image<T>& image) {
        // allocate the parent image and buckets for the tuples
        AreaRules<U> area;
        Parents parents(Parents::infinity, image.width(), image.height());

        // MPI administration
        MPI_Comm_split(this->comm_, 0, this->size_ - this->rank_ - 1, &this->reverse_comm_);
        Tuple<T, U>::create_mpi_type(&this->tuple_type_);
        AreaEndpoint<U>::create_mpi_type(&this->area_endpoint_type_);
        this->create_stitch_op<T, U>(area, parents);
        this->create_area_connect_op<U>();
        this->create_root_find_op<T, U>();

        // calculate the global pixel offset
        U offset = image.size() - image.width();
        MPI_Exscan(MPI_IN_PLACE, &offset, 1, MPI_Types<U>::map(), MPI_SUM, this->comm_);
        offset = this->rank_ == 0 ? 0 : offset;

        // local max tree computation
        MaxTree::compute(image, parents, 4);//this->thread_count_); //TODO
        // distributed resolution
        TupleBuckets<T, U> root_buckets = this->get_halo_roots(image, parents, offset);
        TupleBuckets<T, U> area_buckets = this->connect_halos(image, parents, offset, area);
        this->resolve_tuples(area_buckets, root_buckets);

        // send tuples to owners and apply
        Tuples<T, U> resolved_area, resolved_roots;
        this->redistribute_tuples(image, area_buckets, resolved_area);
        this->redistribute_tuples(image, root_buckets, resolved_roots);
        this->generate_parent_image(parents, resolved_area, resolved_roots, area, offset);

        // MPI clean up
        this->free_root_find_op<T, U>();
        this->free_area_connect_op<U>();
        this->free_stitch_op<T, U>();
        AreaEndpoint<U>::free_mpi_type(&this->area_endpoint_type_);
        Tuple<T, U>::free_mpi_type(&this->tuple_type_);
        MPI_Comm_free(&this->reverse_comm_);

        return parents;
    }

protected:
    MPI_Comm comm_;
    MPI_Comm reverse_comm_;

    MPI_Datatype tuple_type_;
    MPI_Datatype area_endpoint_type_;

    MPI_Op stitch_;
    MPI_Op area_connect_;
    MPI_Op root_find_;

    int rank_;
    int size_;

    std::thread::id id_;
    unsigned int thread_count_;

    template <typename U=Parents::type>
    using StitchElement = std::pair<AreaRules<U>*, Parents*>;
    template <typename U=Parents::type>
    using StitchElements = std::unordered_map<std::thread::id, StitchElement<U>>;

    template <typename T, typename U=Parents::type>
    static std::mutex& get_stitch_mutex() {
        static std::mutex stitch_mutex_;
        return stitch_mutex_;
    }

    template <typename T, typename U=Parents::type>
    static StitchElements<U>& get_stitch_elements() {
        static StitchElements<U> stitch_elements_;
        return stitch_elements_;
    }

    template <typename T, typename U=Parents::type>
    void create_stitch_op(AreaRules<U>& area_, Parents& parents_) {
        DistributedMaxTree::get_stitch_mutex<T, U>().lock();
        DistributedMaxTree::get_stitch_elements<T, U>()[this->id_] = std::make_pair(&area_, &parents_);
        DistributedMaxTree::get_stitch_mutex<T, U>().unlock();

        MPI_Op_create([](void* in_, void* out_, int* len, MPI_Datatype*) {
            U* in = reinterpret_cast<U*>(in_);
            U* out = reinterpret_cast<U*>(out_);

            DistributedMaxTree::get_stitch_mutex<T, U>().lock();
            StitchElement<U> content = DistributedMaxTree::get_stitch_elements<T, U>()[std::this_thread::get_id()];
            DistributedMaxTree::get_stitch_mutex<T, U>().unlock();
            AreaRules<U>* area = content.first;
            Parents* parents = content.second;

            // find the mapping rules
            for (int i = static_cast<int>(parents->width()), j = 0; i < *len; ++i, ++j) {
                U left_canonized = MaxTree::canonize(*area, in[i]);
                U right_canonized = MaxTree::canonize(*area, out[j]);
                auto minmax = std::minmax(left_canonized, right_canonized);
                if (minmax.first != minmax.second) (*area)[minmax.second] = minmax.first;
            }

            // apply the mapping rules on the left outer border of the left image and the right outer border of the right image
            for (size_t i = 0; i < parents->width(); ++i) {
                out[i] = MaxTree::canonize(*area, in[i]);
                out[i + parents->width()] = MaxTree::canonize(*area, out[i + parents->width()]);
            }
        }, false, &this->stitch_);
    }

    template<typename U=Parents::type>
    void create_area_connect_op() {
        MPI_Op_create([](void* in_, void* out_, int*, MPI_Datatype*) {
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
        }, false, &this->area_connect_);
    }

    template<typename T, typename U=Parents::type>
    void create_root_find_op() {
        MPI_Op_create([](void* in_, void* out_, int*, MPI_Datatype*) {
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
        }, false, &this->root_find_);
    }

    template <typename T, typename U=Parents::type>
    TupleBuckets<T, U> get_halo_roots(const Image<T>& image, const Parents& parents, size_t offset) {
        TupleBuckets<T, U> roots;
        std::unordered_set<U> seen;

        if (this->rank_ > 0) {
            this->get_one_sided_halo_roots(image, parents, roots, 0, seen, offset);
        }
        if (this->rank_ + 1 < this->size_) {
            this->get_one_sided_halo_roots(image, parents, roots, image.size() - image.width(), seen, offset);
        }

        return roots;
    }

    template <typename T, typename U=Parents::type>
    void get_one_sided_halo_roots(const Image<T>& image, const Parents& parents, TupleBuckets<T, U>& roots, size_t start, std::unordered_set<U>& seen, size_t offset) {
        for (size_t i = start; i != start + image.width(); ++i) {
            U location = (image[i] == image[parents[i]] ? parents[i] : i);
            U parent = parents[location];

            while (seen.find(location) == seen.end() and location != parent) {
                T first_color = image[location];
                T second_color = image[parent];
                roots[first_color].push_back(Tuple<T, U>(first_color, location + offset, second_color, parent + offset));

                seen.insert(location);
                location = parent;
                parent = parents[location];
            }
        }
    }

    template <typename T, typename U=Parents::type>
    static U canonize(RootRules<T, U>& rules, U origin) {
        U destination = origin;
        auto it = rules.find(destination);

        size_t i = 0;
        while (it != rules.end() and it->second.second != destination) {
            destination = it->second.second;
            it = rules.find(destination);
            ++i;
        }
        return destination;
    }

    template<typename T, typename U=Parents::type>
    TupleBuckets<T, U> connect_halos(const Image<T>& image, Parents& parents, size_t global_offset, AreaRules<U>& area) {
        TupleBuckets<T, U> area_buckets;
        if (this->size_ == 1) return area_buckets;

        const size_t width = parents.width();
        const size_t width2 = width * 2;
        std::vector<U> send_buffer(width2);

        // right directed communication of minimum
        for (size_t i = 0, j = image.size() - width; i < width; ++i, ++j) {
            send_buffer[i] = MaxTree::canonical_point(i, image, parents) + global_offset;
            send_buffer[i + width] = MaxTree::canonical_point(j, image, parents) + global_offset;
        }
        MPI_Scan(MPI_IN_PLACE, send_buffer.data(), width2, MPI_Types<U>::map(), this->stitch_, this->comm_);

        // left directed, backwards propagation of right minimum scan results
        for (size_t i = 0, j = image.size() - width; i < width; ++i, ++j) {
            send_buffer[i + width] = MaxTree::canonize(area, MaxTree::canonical_point(i, image, parents) + global_offset);
            send_buffer[i] = MaxTree::canonize(area, MaxTree::canonical_point(j, image, parents) + global_offset);
        }
        MPI_Scan(MPI_IN_PLACE, send_buffer.data(), width2, MPI_Types<U>::map(), this->stitch_, this->reverse_comm_);

        for (const auto& rule : area) {
            if (rule.first < global_offset or rule.first >= global_offset + image.size()) continue;
            T color = image[rule.first - global_offset];
            area_buckets[color].push_back(Tuple<T, U>(color, rule.first, color, rule.second));
            area_buckets[color].push_back(Tuple<T, U>(color, rule.second, color, rule.first));
        }

        return area_buckets;
    }

    template<typename T, typename U=Parents::type>
    void resolve_tuples(TupleBuckets<T, U>& area_buckets, TupleBuckets<T, U>& root_buckets) {
        if (this->size_ == 1) return;
        auto it = root_buckets.rbegin();

        while (true) {
            // check whether main loop is done
            bool all_done = (it == root_buckets.rend());
            MPI_Allreduce(MPI_IN_PLACE, &all_done, 1, MPI_C_BOOL, MPI_LAND, this->comm_);
            if (all_done) break;

            // retrieve the current bucket
            T color = (it != root_buckets.rend() ? it->first : std::numeric_limits<T>::lowest());
            MPI_Allreduce(MPI_IN_PLACE, &color, 1, MPI_Types<T>::map(), MPI_MAX, this->comm_);
            if (it != root_buckets.rend() and color == it->first) ++it;

            // resolve the area chain
            bool unresolved = true;
            Tuples<T, U>& area_bucket = area_buckets[color];
            Tuples<T, U>& root_bucket = root_buckets[color];

            while (unresolved) {
                AreaRules<U> area;
                this->sample_sort(area_bucket);
                this->resolve_area_chain(area_bucket, area);
                unresolved = this->remap_area_tuples(area_bucket, area);
                MPI_Allreduce(MPI_IN_PLACE, &unresolved, 1, MPI_C_BOOL, MPI_LOR, this->comm_); // globally done?
            }
            // resolve the roots
            add_area_tuples_to_roots(root_bucket, area_bucket);
            this->resolve_roots(color, root_buckets, area_buckets);
        }
    }

    template<typename T, typename U=Parents::type>
    void sample_sort(Tuples<T, U>& bucket) {
        // early out for a single core
        if (this->size_ == 1) {
            std::sort(bucket.begin(), bucket.end());
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
        std::sort(bucket.begin(), bucket.end());

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
        std::sort(splitters.begin(), splitters.end());

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
    }

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
    }

    template <typename T, typename U=Parents::type>
    void resolve_area_chain(Tuples<T, U>& tuples, AreaRules<U>& area) {
        for (const auto& tuple : tuples) {
            U from = tuple.from;
            U to = tuple.to;

            if (to > from) std::swap(from, to);
            U canonical_point = MaxTree::canonize(area, from);

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
    }


    template <typename T, typename U=Parents::type>
    AreaEndpoints<U> initialize_area_ends(Tuples<T, U>& tuples, AreaRules<U>& area) {
        AreaEndpoints<U> ends;
        if (tuples.empty()) return ends;

        Tuple<T, U>& front = tuples.front();
        AreaEndpoint<U>& left = ends.front();
        auto left_from_to = std::minmax(front.from, front.to);
        left.from = left_from_to.second;
        left.to = left_from_to.first;
        left.canonical_to = MaxTree::canonize(area, left.from);

        Tuple<T, U>& back = tuples.back();
        AreaEndpoint<U>& right = ends.back();
        auto right_from_to = std::minmax(back.from, back.to);
        right.from = right_from_to.second;
        right.to = right_from_to.first;
        right.canonical_to = MaxTree::canonize(area, right.from);

        return ends;
    }

    template <typename U=Parents::type>
    void stitch_area_ends(AreaEndpoints<U>& ends, AreaRules<U>& area) {
        const AreaEndpoint<U>& left = ends.front();
        const AreaEndpoint<U>& right = ends.back();

        if (left.to != left.canonical_to)
            area[left.to] = left.canonical_to;
        if (right.to != right.canonical_to)
            area[right.to] = right.canonical_to;
    }

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
            U area_root = MaxTree::canonize(area, from);

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
    }

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
    }

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
    }

    template <typename T, typename U=Parents::type>
    void stitch_ends(Endpoints<T, U>& ends, RootRules<T, U>& roots) {
        const Tuple<T, U>& left = ends.front();
        roots[left.from] = Root<T, U>(left.neighbor_color, left.to);

        const Tuple<T, U>& right = ends.back();
        roots[right.from] = Root<T, U>(right.neighbor_color, right.to);
    }

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
                U canonized_to = DistributedMaxTree::canonize(area, tuple.to);
                U canonized_root = DistributedMaxTree::canonize(area, root.second);
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
    }

    template<typename T, typename U=Parents::type>
    void redistribute_tuples(const Image<T>& image, TupleBuckets<T, U>& buckets, Tuples<T, U>& resolved) {
        if (this->size_ == 1) {
            for (auto& bucket : buckets) {
                resolved.insert(resolved.end(), bucket.second.begin(), bucket.second.end());
                bucket.second.clear();
            }
            return;
        }

        // determine the total number of pixels in the image
        size_t total_height = image.height() - (this->rank_ + 1 != this->size_ ? 1 : 0);
        MPI_Allreduce(MPI_IN_PLACE, &total_height, 1, MPI_Types<decltype(total_height)>::map(), MPI_SUM, this->comm_);

        // chunk up the image to determine tuple target ranks
        U chunk = total_height / this->size_ * image.width();
        U remainder = total_height % this->size_;
        U total_tuples = 0;

        // calculate which tuple goes where
        std::vector<int> send_counts(static_cast<size_t>(this->size_), 0);
        std::vector<int> recv_counts(static_cast<size_t>(this->size_), 0);
        for (const auto& b : buckets) {
            auto& bucket = b.second;
            for (const auto& tuple : bucket) {
                U target_rank = tuple.from / chunk;
                while (target_rank != 0 and target_rank * chunk + std::min(target_rank, remainder) * image.width() > tuple.from) {
                    --target_rank;
                }
                ++send_counts[target_rank];
                ++total_tuples;

                if (target_rank > 0 and target_rank * chunk + (std::min(target_rank, remainder) + 1) * image.width() > tuple.from) {
                    ++send_counts[target_rank - 1];
                    ++total_tuples;
                }
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

        for (auto& b : buckets) {
            auto& bucket = b.second;
            for (const auto& tuple : bucket) {
                U target_rank = tuple.from / chunk;
                while (target_rank != 0 and target_rank * chunk + std::min(target_rank, remainder) * image.width() > tuple.from) {
                    --target_rank;
                }
                int& position = placement[target_rank];
                outgoing[position] = tuple;
                ++position;

                if (target_rank > 0 and target_rank * chunk + (std::min(target_rank, remainder) + 1) * image.width() > tuple.from) {
                    int& position = placement[target_rank - 1];
                    outgoing[position] = tuple;
                    ++position;
                }
            }
            bucket.clear();
        }

        MPI_Alltoallv(outgoing.data(), send_counts.data(), send_displs.data(), this->tuple_type_,
                      resolved.data(), recv_counts.data(), recv_displs.data(), this->tuple_type_, this->comm_);
    }

    template<typename T, typename U=Parents::type>
    void generate_parent_image(Parents& parents, Tuples<T, U>& resolved_area, Tuples<T, U>& resolved_roots, AreaRules<U>& area, size_t offset) {
        // early out for one node
        if (this->size_ == 1) return;

        // actually parse the area tuples first
        area.clear();
        for (const auto& tuple : resolved_area) {
            if (tuple.from < tuple.to) continue;
            area[tuple.from] = tuple.to;
        }

        // canonize the flat areas
        for (auto& pixel : parents) {
            pixel += offset;
            pixel = MaxTree::canonize(area, pixel);
        }

        // canonize the former area roots
        for (auto& rule : area) {
            parents[rule.first - offset] = rule.second;
        }

        // and finally set the roots
        for (const auto& root : resolved_roots) {
            parents[root.from - offset] = root.to;
        }
    }

    template <typename T, typename U=Parents::type>
    void free_stitch_op() {
        DistributedMaxTree::get_stitch_mutex<T, U>().lock();
        DistributedMaxTree::get_stitch_elements<T, U>().erase(this->id_);
        DistributedMaxTree::get_stitch_mutex<T, U>().unlock();
    }

    template <typename U=Parents::type>
    void free_area_connect_op() {
        MPI_Op_free(&this->area_connect_);
    }

    template <typename T, typename U=Parents::type>
    void free_root_find_op() {
        MPI_Op_free(&this->root_find_);
    }
};

#endif // DISTRIBUTED_MAX_TREE_H