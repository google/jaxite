#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <random>

#include "third_party/tensorflow/compiler/xla/ffi/api/c_api.h"
#include "third_party/tensorflow/compiler/xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
#define COORDINATE_NUM 4
#define CHUNK_NUM 32
#define RESERVE_RATIO 2.0
#define SCALAR_BITS 253
#define ORDER_HIGH 0x12AB655E
#define ORDER_LOW_BITS 224
// #define PROFILE

typedef struct {
  uint32_t chunks[CHUNK_NUM * COORDINATE_NUM];
} point_t;

typedef struct {
  // Parameters for the distributor
  uint32_t window_num;
  uint32_t regular_bucket_num;
  uint32_t special_bucket_num;
  uint32_t msm_length;
  uint32_t fixed_regular_padding_size;
  uint32_t fixed_special_padding_size;

  // Inputs buffers
  uint32_t* zero;
  int32_t* slices_list;
  uint32_t* points_list;
  uint32_t* neg_points_list;

  // Output buffers
  uint32_t* regular_buckets;
  uint32_t* special_buckets;
} distributor_params_t;

class Distributor {
 public:
  Distributor(uint32_t window_num, uint32_t regular_bucket_num,
              uint32_t special_bucket_num, uint32_t msm_length)
      : window_num_(window_num),
        regular_bucket_num_(regular_bucket_num),
        special_bucket_num_(special_bucket_num),
        msm_length_(msm_length) {
    this->reserve_ratio_ = RESERVE_RATIO;
    this->windows_.resize(window_num);
    this->fixed_regular_padding_size_ = 0;
    this->fixed_special_padding_size_ = 0;
    this->truncated = 0;
  }

  void set_slices_list(int32_t* slices_list) {
    this->slices_list_pointers_.resize(this->window_num_);
    for (uint32_t i = 0; i < this->window_num_; ++i) {
      slices_list_pointers_[i] = slices_list + i * this->msm_length_;
    }
  }

  void set_output_buffers(uint32_t* regular_buckets,
                          uint32_t* special_buckets) {
    this->regular_buckets_ = regular_buckets;
    this->special_buckets_ = special_buckets;
    this->bucket_pointers_.resize(window_num_);
    this->bucket_sizes_.resize(window_num_);
    uint32_t regular_bucket_capacity_in_bytes =
        fixed_regular_padding_size_ * sizeof(point_t);
    uint32_t special_bucket_capacity_in_bytes =
        fixed_special_padding_size_ * sizeof(point_t);
    uint32_t regular_window_capacity_in_bytes =
        regular_bucket_num_ * regular_bucket_capacity_in_bytes;
    // uint32_t special_window_capacity_in_bytes = special_bucket_num_ *
    // special_bucket_capacity_in_bytes;

    // Initialize the bucket pointers and sizes
    for (uint32_t i = 0; i < window_num_ - 1; ++i) {
      this->bucket_pointers_[i].resize(regular_bucket_num_);
      this->bucket_sizes_[i].resize(regular_bucket_num_);
      for (uint32_t j = 0; j < this->bucket_pointers_[i].size(); ++j) {
        this->bucket_sizes_[i][j] = 0;
        this->bucket_pointers_[i][j] = reinterpret_cast<point_t*>(
            reinterpret_cast<uint8_t*>(regular_buckets_) +
            i * regular_window_capacity_in_bytes +
            j * regular_bucket_capacity_in_bytes);
      }
    }
    // For the last window, which has special buckets
    this->bucket_pointers_[window_num_ - 1].resize(special_bucket_num_);
    this->bucket_sizes_[window_num_ - 1].resize(special_bucket_num_);
    for (uint32_t j = 0; j < this->bucket_pointers_[window_num_ - 1].size();
         ++j) {
      this->bucket_sizes_[window_num_ - 1][j] = 0;
      this->bucket_pointers_[window_num_ - 1][j] = reinterpret_cast<point_t*>(
          reinterpret_cast<uint8_t*>(special_buckets_) +
          j * special_bucket_capacity_in_bytes);
    }
  }

  void set_points_list(uint32_t* points_list) {
    this->points_list_ = reinterpret_cast<point_t*>(points_list);
  }

  void set_neg_points_list(uint32_t* neg_points_list) {
    this->neg_points_list_ = reinterpret_cast<point_t*>(neg_points_list);
  }

  void set_zeros(uint32_t* zero) {
    this->zero_ = reinterpret_cast<point_t*>(zero);
  }

  void set_fixed_padding_size(uint32_t regular_size, uint32_t special_size) {
    this->fixed_regular_padding_size_ = regular_size;
    this->fixed_special_padding_size_ = special_size;
  }

  void set_tile_length(uint32_t tile_length) {
    this->tile_length_ = tile_length;
    assert(msm_length_ % tile_length == 0);
    this->tile_num_ = msm_length_ / tile_length;
  }

  void distribute_a_window(int32_t* slices, point_t* points,
                           uint32_t bucket_num,
                           std::vector<std::vector<point_t>>& buckets,
                           uint32_t& max_size) {
    // Implement the distribution logic for a single window here
    buckets.resize(bucket_num);
    uint32_t reserve_num =
        static_cast<uint32_t>(msm_length_ * reserve_ratio_ / bucket_num);
    for (uint32_t i = 0; i < bucket_num; ++i) {
      buckets[i].reserve(reserve_num);
    }

    for (uint32_t i = 0; i < msm_length_; ++i) {
      if (slices[i] == 0) {
        continue;
      }
      uint32_t bucket_index = slices[i] - 1;
      buckets[bucket_index].push_back(points[i]);
    }
    // max_size = 0;
    for (uint32_t i = 0; i < bucket_num; ++i) {
      if (buckets[i].size() > max_size) {
        max_size = buckets[i].size();
      }
    }
  }

  void distribute_a_window_to_buffer(int32_t* slices, point_t* points,
                                     uint32_t tile_length,
                                     uint32_t max_bucket_size,
                                     std::vector<point_t*>& buckets,
                                     std::vector<uint32_t>& bucket_sizes,
                                     uint32_t& overflow) {
    for (uint32_t i = 0; i < tile_length; ++i) {
      uint32_t prefetch_distance = 32;  // Adjust prefetch distance as needed
      if (i + prefetch_distance < tile_length) {
        __builtin_prefetch(&slices[i + prefetch_distance], 0, 1);
        __builtin_prefetch(&points[i + prefetch_distance], 0, 1);
      }
      if (slices[i] == 0) {
        continue;
      }
      int32_t bucket_index = slices[i] - 1;
      if (bucket_index >= buckets.size()) {
        std::cerr << "Error: bucket index out of range." << std::endl;
        continue;
      }
      if (bucket_sizes[bucket_index] < max_bucket_size) {
        buckets[bucket_index][bucket_sizes[bucket_index]] = points[i];
        bucket_sizes[bucket_index]++;
      } else {
        overflow++;
      }
    }
  }

  void distribute_a_window_to_buffer_with_sign(
      int32_t* slices, point_t* points, point_t* neg_points,
      uint32_t tile_length, uint32_t max_bucket_size,
      std::vector<point_t*>& buckets, std::vector<uint32_t>& bucket_sizes,
      uint32_t& overflow) {
    for (uint32_t i = 0; i < tile_length; ++i) {
      if (slices[i] == 0) {
        continue;
      }
      point_t* points_to_use = (slices[i] > 0) ? &points[i] : &neg_points[i];
      int32_t bucket_index = abs(slices[i]) - 1;
      if (bucket_index >= buckets.size()) {
        std::cerr << "Error: bucket index out of range." << std::endl;
        continue;
      }
      if (bucket_sizes[bucket_index] < max_bucket_size) {
        buckets[bucket_index][bucket_sizes[bucket_index]] = *points_to_use;
        bucket_sizes[bucket_index]++;
      } else {
        overflow++;
      }
    }
  }

  void pad_a_window(point_t* zero, uint32_t target_size,
                    std::vector<std::vector<point_t>>& buckets) {
    for (uint32_t i = 0; i < buckets.size(); ++i) {
      uint32_t current_size = buckets[i].size();
      // assert(current_size <= target_size);
      if (current_size <= target_size) {
        uint32_t pad_num = target_size - current_size;
        buckets[i].insert(buckets[i].end(), pad_num, *zero);
      } else {
        // If the current size exceeds the target size, truncate it
        buckets[i].resize(target_size);
      }
    }
  }

  void pad_a_window_to_buffer(point_t* zero, uint32_t target_size,
                              std::vector<point_t*>& buckets,
                              std::vector<uint32_t>& bucket_sizes) {
    for (uint32_t i = 0; i < buckets.size(); ++i) {
      uint32_t current_size = bucket_sizes[i];
      // assert(current_size <= target_size);
      if (current_size < target_size) {
        uint32_t pad_num = target_size - current_size;
        for (uint32_t j = 0; j < pad_num; ++j) {
          buckets[i][current_size + j] = *zero;
        }
        bucket_sizes[i] += pad_num;
      }
    }
  }

  void distribute() {
    uint32_t max_regular_bucket_size = 0;
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < window_num_ - 1; ++i) {
      int32_t* slices = slices_list_pointers_[i];
      distribute_a_window(slices, points_list_, regular_bucket_num_,
                          windows_[i], max_regular_bucket_size);
    }
    uint32_t max_special_bucket_size = 0;
    auto end2 = std::chrono::high_resolution_clock::now();
    distribute_a_window(slices_list_pointers_[window_num_ - 1], points_list_,
                        special_bucket_num_, windows_[window_num_ - 1],
                        max_special_bucket_size);
    end = std::chrono::high_resolution_clock::now();

    duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    auto duration2_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end2 - start)
            .count();
    std::cout << "dist_r time: " << duration2_us << " us" << std::endl;
    std::cout << "dist time: " << duration_us << " us" << std::endl;

    this->real_regular_padding_size_ = max_regular_bucket_size;
    this->real_special_padding_size_ = max_special_bucket_size;

    if (this->fixed_regular_padding_size_ > 0) {
      if (this->fixed_regular_padding_size_ < max_regular_bucket_size) {
        this->truncated = 1;
      }
      // assert(this->fixed_regular_padding_size_ >= max_regular_bucket_size);
      max_regular_bucket_size = this->fixed_regular_padding_size_;
    }
    if (this->fixed_special_padding_size_ > 0) {
      if (this->fixed_special_padding_size_ < max_special_bucket_size) {
        this->truncated = 1;
      }
      // assert(this->fixed_special_padding_size_ >= max_special_bucket_size);
      max_special_bucket_size = this->fixed_special_padding_size_;
    }

    start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < window_num_; ++i) {
      uint32_t target_size = (i == window_num_ - 1) ? max_special_bucket_size
                                                    : max_regular_bucket_size;
      pad_a_window(zero_, target_size, windows_[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "pad time: " << duration_us << " us" << std::endl;
  }

  void distribute_parallel_v1() {
    uint32_t max_regular_bucket_size = 0;
    std::vector<uint32_t> max_regular_bucket_sizes(window_num_ - 1, 0);
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (uint32_t i = 0; i < window_num_ - 1; ++i) {
      int32_t* slices = slices_list_pointers_[i];
      distribute_a_window(slices, points_list_, regular_bucket_num_,
                          windows_[i], max_regular_bucket_sizes[i]);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < window_num_ - 1; ++i) {
      if (max_regular_bucket_sizes[i] > max_regular_bucket_size) {
        max_regular_bucket_size = max_regular_bucket_sizes[i];
      }
    }
    uint32_t max_special_bucket_size = 0;

    distribute_a_window(slices_list_pointers_[window_num_ - 1], points_list_,
                        special_bucket_num_, windows_[window_num_ - 1],
                        max_special_bucket_size);
    this->real_regular_padding_size_ = max_regular_bucket_size;
    this->real_special_padding_size_ = max_special_bucket_size;
    end = std::chrono::high_resolution_clock::now();

    duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    auto duration2_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end2 - start)
            .count();
    std::cout << "dist_r time: " << duration2_us << " us" << std::endl;
    std::cout << "dist time: " << duration_us << " us" << std::endl;

    if (this->fixed_regular_padding_size_ > 0) {
      if (this->fixed_regular_padding_size_ < max_regular_bucket_size) {
        this->truncated = 1;
      }
      // assert(this->fixed_regular_padding_size_ >= max_regular_bucket_size);
      max_regular_bucket_size = this->fixed_regular_padding_size_;
    }
    if (this->fixed_special_padding_size_ > 0) {
      if (this->fixed_special_padding_size_ < max_special_bucket_size) {
        this->truncated = 1;
      }
      // assert(this->fixed_special_padding_size_ >=
      // max_special_bucket_size);
      max_special_bucket_size = this->fixed_special_padding_size_;
    }

    start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (uint32_t i = 0; i < window_num_; ++i) {
      uint32_t target_size = (i == window_num_ - 1) ? max_special_bucket_size
                                                    : max_regular_bucket_size;
      pad_a_window(zero_, target_size, windows_[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "pad time: " << duration_us << " us" << std::endl;
  }

  void distribute_a_window_to_buffer_zero_indexed(
      int32_t* slices, point_t* points, uint32_t tile_length,
      uint32_t max_bucket_size, std::vector<point_t*>& buckets,
      std::vector<uint32_t>& bucket_sizes, uint32_t& overflow) {
    for (uint32_t i = 0; i < tile_length; ++i) {
      uint32_t prefetch_distance = 32;
      if (i + prefetch_distance < tile_length) {
        __builtin_prefetch(&slices[i + prefetch_distance], 0, 1);
        __builtin_prefetch(&points[i + prefetch_distance], 0, 1);
      }
      if (slices[i] <= 0) {
        continue;
      }
      int32_t bucket_index = slices[i];
      if (bucket_index >= static_cast<int32_t>(buckets.size())) {
        std::cerr << "Error: bucket index out of range." << std::endl;
        continue;
      }
      if (bucket_sizes[bucket_index] < max_bucket_size) {
        buckets[bucket_index][bucket_sizes[bucket_index]] = points[i];
        bucket_sizes[bucket_index]++;
      } else {
        overflow++;
      }
    }
  }

  void distribute_to_buffer_zero_indexed_parallel() {
    std::vector<uint32_t> bucket_overflow(window_num_, 0);
#ifdef PROFILE
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    start = std::chrono::high_resolution_clock::now();
#endif
#pragma omp parallel for
    for (uint32_t i = 0; i < window_num_ - 1; ++i) {
      int32_t* slices = slices_list_pointers_[i];
      point_t* points = points_list_;
      std::vector<point_t*>& buckets = bucket_pointers_[i];
      std::vector<uint32_t>& bucket_sizes = bucket_sizes_[i];
      uint32_t max_bucket_size = this->fixed_regular_padding_size_;
      distribute_a_window_to_buffer_zero_indexed(
          slices, points, msm_length_, max_bucket_size, buckets, bucket_sizes,
          bucket_overflow[i]);
    }
#ifdef PROFILE
    auto end2 = std::chrono::high_resolution_clock::now();
#endif

    distribute_a_window_to_buffer_zero_indexed(
        slices_list_pointers_[window_num_ - 1], points_list_, msm_length_,
        this->fixed_special_padding_size_, bucket_pointers_[window_num_ - 1],
        bucket_sizes_[window_num_ - 1], bucket_overflow[window_num_ - 1]);

#ifdef PROFILE
    end = std::chrono::high_resolution_clock::now();
    duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    auto duration2_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end2 - start)
            .count();
    std::cout << "dist_r time: " << duration2_us << " us" << std::endl;
    std::cout << "dist time: " << duration_us << " us" << std::endl;
#endif

    for (uint32_t i = 0; i < window_num_; ++i) {
      if (bucket_overflow[i] > 0) {
        this->truncated = 1;
        break;
      }
    }

#ifdef PROFILE
    start = std::chrono::high_resolution_clock::now();
#endif
#pragma omp parallel for
    for (uint32_t i = 0; i < window_num_; ++i) {
      uint32_t target_size = (i == window_num_ - 1)
                                 ? this->fixed_special_padding_size_
                                 : this->fixed_regular_padding_size_;
      pad_a_window_to_buffer(zero_, target_size, bucket_pointers_[i],
                             bucket_sizes_[i]);
    }
#ifdef PROFILE
    end = std::chrono::high_resolution_clock::now();
    duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "pad time: " << duration_us << " us" << std::endl;
#endif
  }

  void distribute_to_buffer_parallel() {
    std::vector<uint32_t> bucket_overflow(window_num_, 0);
#ifdef PROFILE
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    start = std::chrono::high_resolution_clock::now();
#endif
#pragma omp parallel for
    for (uint32_t i = 0; i < window_num_ - 1; ++i) {
      int32_t* slices = slices_list_pointers_[i];
      point_t* points = points_list_;
      std::vector<point_t*>& buckets = bucket_pointers_[i];
      std::vector<uint32_t>& bucket_sizes = bucket_sizes_[i];
      uint32_t max_bucket_size = this->fixed_regular_padding_size_;
      distribute_a_window_to_buffer(slices, points, msm_length_,
                                    max_bucket_size, buckets, bucket_sizes,
                                    bucket_overflow[i]);
    }
#ifdef PROFILE
    auto end2 = std::chrono::high_resolution_clock::now();
#endif

    distribute_a_window_to_buffer(
        slices_list_pointers_[window_num_ - 1], points_list_, msm_length_,
        this->fixed_special_padding_size_, bucket_pointers_[window_num_ - 1],
        bucket_sizes_[window_num_ - 1], bucket_overflow[window_num_ - 1]);

#ifdef PROFILE
    end = std::chrono::high_resolution_clock::now();
    duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    auto duration2_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end2 - start)
            .count();
    std::cout << "dist_r time: " << duration2_us << " us" << std::endl;
    std::cout << "dist time: " << duration_us << " us" << std::endl;
#endif

    for (uint32_t i = 0; i < window_num_; ++i) {
      if (bucket_overflow[i] > 0) {
        this->truncated = 1;
        break;
      }
    }

#ifdef PROFILE
    start = std::chrono::high_resolution_clock::now();
#endif
#pragma omp parallel for
    for (uint32_t i = 0; i < window_num_; ++i) {
      uint32_t target_size = (i == window_num_ - 1)
                                 ? this->fixed_special_padding_size_
                                 : this->fixed_regular_padding_size_;
      pad_a_window_to_buffer(zero_, target_size, bucket_pointers_[i],
                             bucket_sizes_[i]);
    }
#ifdef PROFILE
    end = std::chrono::high_resolution_clock::now();
    duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "pad time: " << duration_us << " us" << std::endl;
#endif
  }

  void distribute_to_buffer_signed_parallel() {
    std::vector<uint32_t> bucket_overflow(window_num_, 0);
#ifdef PROFILE
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    start = std::chrono::high_resolution_clock::now();
#endif
#pragma omp parallel for
    for (uint32_t i = 0; i < window_num_ - 1; ++i) {
      int32_t* slices = slices_list_pointers_[i];
      point_t* points = points_list_;
      point_t* neg_points = neg_points_list_;
      std::vector<point_t*>& buckets = bucket_pointers_[i];
      std::vector<uint32_t>& bucket_sizes = bucket_sizes_[i];
      uint32_t max_bucket_size = this->fixed_regular_padding_size_;
      distribute_a_window_to_buffer_with_sign(
          slices, points, neg_points, msm_length_, max_bucket_size, buckets,
          bucket_sizes, bucket_overflow[i]);
    }
#ifdef PROFILE
    auto end2 = std::chrono::high_resolution_clock::now();
#endif

    distribute_a_window_to_buffer_with_sign(
        slices_list_pointers_[window_num_ - 1], points_list_, neg_points_list_,
        msm_length_, this->fixed_special_padding_size_,
        bucket_pointers_[window_num_ - 1], bucket_sizes_[window_num_ - 1],
        bucket_overflow[window_num_ - 1]);

#ifdef PROFILE
    end = std::chrono::high_resolution_clock::now();
    duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    auto duration2_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end2 - start)
            .count();
    std::cout << "dist_r time: " << duration2_us << " us" << std::endl;
    std::cout << "dist time: " << duration_us << " us" << std::endl;
#endif

    for (uint32_t i = 0; i < window_num_; ++i) {
      if (bucket_overflow[i] > 0) {
        this->truncated = 1;
        break;
      }
    }

#ifdef PROFILE
    start = std::chrono::high_resolution_clock::now();
#endif
#pragma omp parallel for
    for (uint32_t i = 0; i < window_num_; ++i) {
      uint32_t target_size = (i == window_num_ - 1)
                                 ? this->fixed_special_padding_size_
                                 : this->fixed_regular_padding_size_;
      pad_a_window_to_buffer(zero_, target_size, bucket_pointers_[i],
                             bucket_sizes_[i]);
    }
#ifdef PROFILE
    end = std::chrono::high_resolution_clock::now();
    duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "pad time: " << duration_us << " us" << std::endl;
#endif
  }

  void distribute_to_buffer_parallel_v2() {
    std::vector<uint32_t> bucket_overflow(window_num_, 0);
#ifdef PROFILE
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    start = std::chrono::high_resolution_clock::now();
#endif
    for (uint32_t t = 0; t < tile_num_; ++t) {
// Distribute each tile to the corresponding window
#pragma omp parallel for
      for (uint32_t i = 0; i < window_num_ - 1; ++i) {
        int32_t* slices = slices_list_pointers_[i] + t * tile_length_;
        point_t* points = points_list_ + t * tile_length_;
        std::vector<point_t*>& buckets = bucket_pointers_[i];
        std::vector<uint32_t>& bucket_sizes = bucket_sizes_[i];
        uint32_t max_bucket_size = this->fixed_regular_padding_size_;
        distribute_a_window_to_buffer(slices, points, tile_length_,
                                      max_bucket_size, buckets, bucket_sizes,
                                      bucket_overflow[i]);
      }
    }
#ifdef PROFILE
    auto end2 = std::chrono::high_resolution_clock::now();
#endif

    distribute_a_window_to_buffer(
        slices_list_pointers_[window_num_ - 1], points_list_, msm_length_,
        this->real_special_padding_size_, bucket_pointers_[window_num_ - 1],
        bucket_sizes_[window_num_ - 1], bucket_overflow[window_num_ - 1]);

#ifdef PROFILE
    end = std::chrono::high_resolution_clock::now();
    duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    auto duration2_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end2 - start)
            .count();
    std::cout << "dist_r time: " << duration2_us << " us" << std::endl;
    std::cout << "dist time: " << duration_us << " us" << std::endl;
#endif

    for (uint32_t i = 0; i < window_num_; ++i) {
      if (bucket_overflow[i] > 0) {
        this->truncated = 1;
        break;
      }
    }

#ifdef PROFILE
    start = std::chrono::high_resolution_clock::now();
#endif
#pragma omp parallel for
    for (uint32_t i = 0; i < window_num_; ++i) {
      uint32_t target_size = (i == window_num_ - 1)
                                 ? this->fixed_special_padding_size_
                                 : this->fixed_regular_padding_size_;
      pad_a_window_to_buffer(zero_, target_size, bucket_pointers_[i],
                             bucket_sizes_[i]);
    }
#ifdef PROFILE
    end = std::chrono::high_resolution_clock::now();
    duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "pad time: " << duration_us << " us" << std::endl;
#endif
  }

  void get_merged_regular_buckets(uint32_t* regular_buckets) {
    uint32_t offset = 0;
    for (uint32_t i = 0; i < window_num_ - 1; ++i) {
      uint32_t bucket_size = windows_[i].size();
      for (uint32_t j = 0; j < bucket_size; ++j) {
        std::memcpy(regular_buckets + offset, windows_[i][j].data(),
                    windows_[i][j].size() * sizeof(point_t));
        offset += windows_[i][j].size() * CHUNK_NUM * COORDINATE_NUM;
      }
    }
  }

  void get_merged_special_buckets(uint32_t* special_buckets) {
    uint32_t offset = 0;
    uint32_t window_idx = window_num_ - 1;
    uint32_t bucket_size = windows_[window_num_ - 1].size();
    for (uint32_t j = 0; j < bucket_size; ++j) {
      std::memcpy(special_buckets + offset, windows_[window_idx][j].data(),
                  windows_[window_idx][j].size() * sizeof(point_t));
      offset += windows_[window_idx][j].size() * CHUNK_NUM * COORDINATE_NUM;
    }
  }

  void get_merged_metadata(uint32_t* metadata) {
    metadata[0] = real_regular_padding_size_;
    metadata[1] = real_special_padding_size_;
  }

 private:
  uint32_t window_num_;
  uint32_t regular_bucket_num_;
  uint32_t special_bucket_num_;
  uint32_t msm_length_;

  uint32_t tile_length_;
  uint32_t tile_num_;
  float reserve_ratio_;

  std::vector<int32_t*> slices_list_pointers_;
  point_t* points_list_;
  point_t* neg_points_list_;
  point_t* zero_;
  std::vector<std::vector<std::vector<point_t>>> windows_;

  uint32_t fixed_regular_padding_size_;
  uint32_t real_regular_padding_size_;
  uint32_t fixed_special_padding_size_;
  uint32_t real_special_padding_size_;

  uint32_t* regular_buckets_;
  uint32_t* special_buckets_;
  // windows<buckets>
  std::vector<std::vector<point_t*>> bucket_pointers_;
  std::vector<std::vector<uint32_t>> bucket_sizes_;

  /*When truncated is true, the reuslt will be incorrect.
   It is only for perfmance profiling goal.*/
  uint32_t truncated;

  /* For profiling*/
};

distributor_params_t init_distributor_param(uint32_t slice_length,
                                            uint32_t msm_length,
                                            double buf_extend_ratio,
                                            bool signed_bucket) {
  // General parameter initialization
  uint32_t window_num = (SCALAR_BITS + slice_length - 1) / slice_length;
  uint32_t regular_bucket_num = (1 << slice_length) - 1;  // 2^slice_length - 1
  uint32_t shift_bits = ((window_num - 1) * slice_length) - ORDER_LOW_BITS;
  assert(shift_bits >= 0);
  uint32_t special_bucket_num = ORDER_HIGH >> shift_bits;
  if (signed_bucket) {
    regular_bucket_num = 1 << (slice_length - 1);  // 2^(slice_length - 1)
    special_bucket_num++;
    assert(special_bucket_num <= regular_bucket_num);
  }

  // Special bucket optoimization parameter initialization
  uint32_t log_special_duplication_ratio = static_cast<uint32_t>(std::ceil(
      std::log2(static_cast<double>(regular_bucket_num) / special_bucket_num)));
  uint32_t special_duplication_ratio = 1U << log_special_duplication_ratio;
  uint32_t bucket_num_duplication =
      special_bucket_num * special_duplication_ratio;

  // Output buffer parameter initialization
  double expected_regular_bucket_size =
      static_cast<double>(msm_length) / (regular_bucket_num + 1);
  double expected_special_bucket_size =
      static_cast<double>(msm_length) / bucket_num_duplication;
  uint32_t regular_bucket_size =
      static_cast<uint32_t>(expected_regular_bucket_size * buf_extend_ratio);
  uint32_t special_bucket_size =
      static_cast<uint32_t>(expected_special_bucket_size * buf_extend_ratio);
  uint32_t regular_buffer_size_in_U32 = (window_num - 1) * regular_bucket_num *
                                        regular_bucket_size * COORDINATE_NUM *
                                        CHUNK_NUM;
  uint32_t special_buffer_size_in_U32 =
      special_bucket_num * special_duplication_ratio * special_bucket_size *
      COORDINATE_NUM * CHUNK_NUM;

  distributor_params_t params;
  params.window_num = window_num;
  params.regular_bucket_num = regular_bucket_num;
  params.special_bucket_num = special_bucket_num;
  params.msm_length = msm_length;
  params.fixed_regular_padding_size = regular_bucket_size;
  params.fixed_special_padding_size =
      special_bucket_size * special_duplication_ratio;
  params.zero =
      new uint32_t[COORDINATE_NUM * CHUNK_NUM]();  // Initialize to zero
  params.slices_list = new int32_t[window_num * msm_length];
  params.points_list = new uint32_t[msm_length * COORDINATE_NUM * CHUNK_NUM];
  params.neg_points_list =
      new uint32_t[msm_length * COORDINATE_NUM * CHUNK_NUM];
  params.regular_buckets = new uint32_t[regular_buffer_size_in_U32];
  params.special_buckets = new uint32_t[special_buffer_size_in_U32];
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> regular_dist(0, regular_bucket_num);
  std::uniform_int_distribution<uint32_t> special_dist(0, special_bucket_num);
  // Initialize slices_list with random values
  for (uint32_t w = 0; w < window_num - 1; ++w) {
    for (uint32_t i = 0; i < msm_length; ++i) {
      params.slices_list[w * msm_length + i] = regular_dist(gen);
    }
  }
  for (uint32_t i = 0; i < msm_length; ++i) {
    params.slices_list[(window_num - 1) * msm_length + i] = special_dist(gen);
  }

  return params;
}

void free_distributor_param(distributor_params_t& params) {
  delete[] params.zero;
  delete[] params.slices_list;
  delete[] params.points_list;
  delete[] params.neg_points_list;
  delete[] params.regular_buckets;
  delete[] params.special_buckets;
}

int main_old() {
  // Example usage of the Distributor class
  uint32_t slice_length = 10;     // Example slice length
  uint32_t msm_length = 1 << 16;  // Example MSM length
  double buf_extend_ratio = 1.1;  // Example buffer extend ratio
  bool signed_bucket = false;     // Example signed bucket flag
  uint32_t repeat = 50;
  double duration_us = 0;
  // printing the parameters
  std::cout << "Slice Length: " << slice_length << std::endl;
  std::cout << "MSM Length: " << msm_length << std::endl;
  std::cout << "Buffer Extend Ratio: " << buf_extend_ratio << std::endl;
  std::cout << "Signed Bucket: " << (signed_bucket ? "true" : "false")
            << std::endl;

  for (uint32_t i = 0; i < repeat; ++i) {
    std::cout << "Run " << i + 1 << " of " << repeat << std::endl;
    distributor_params_t params = init_distributor_param(
        slice_length, msm_length, buf_extend_ratio, signed_bucket);

    auto start_time = std::chrono::high_resolution_clock::now();

    Distributor* distributor =
        new Distributor(params.window_num, params.regular_bucket_num,
                        params.special_bucket_num, params.msm_length);

    distributor->set_fixed_padding_size(params.fixed_regular_padding_size,
                                        params.fixed_special_padding_size);
    distributor->set_slices_list(params.slices_list);
    distributor->set_points_list(params.points_list);
    distributor->set_neg_points_list(params.neg_points_list);
    distributor->set_zeros(params.zero);
    distributor->set_output_buffers(params.regular_buckets,
                                    params.special_buckets);

    distributor->distribute_to_buffer_parallel();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_time - start_time)
                        .count();
    std::cout << "Distribution time: " << duration << " us" << std::endl;
    duration_us += duration;

    // Free allocated memory
    free_distributor_param(params);
    delete distributor;
  }
  std::cout << "Average distribution time: " << (duration_us / repeat) << " us"
            << std::endl;

  return 0;
}

// Dummy performance test function (replace with your real test)
double run_performance_test(uint32_t slice_length, uint32_t msm_length,
                            double buf_extend_ratio, bool signed_bucket) {
  distributor_params_t params = init_distributor_param(
      slice_length, msm_length, buf_extend_ratio, signed_bucket);

  auto start_time = std::chrono::high_resolution_clock::now();

  Distributor distributor =
      Distributor(params.window_num, params.regular_bucket_num,
                  params.special_bucket_num, params.msm_length);

  distributor.set_fixed_padding_size(params.fixed_regular_padding_size,
                                     params.fixed_special_padding_size);
  distributor.set_slices_list(params.slices_list);
  distributor.set_points_list(params.points_list);
  distributor.set_neg_points_list(params.neg_points_list);
  distributor.set_zeros(params.zero);
  distributor.set_output_buffers(params.regular_buckets,
                                 params.special_buckets);

  distributor.distribute_to_buffer_parallel();
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                      end_time - start_time)
                      .count();
  // std::cout << "Distribution time: " << duration << " us" << std::endl;
  // Free allocated memory
  free_distributor_param(params);
  return static_cast<double>(duration);
}

// Compute statistics
void compute_stats(const std::vector<double>& data, double& min, double& max,
                   double& avg, double& stddev) {
  if (data.empty()) {
    min = max = avg = stddev = 0.0;
    return;
  }
  min = *std::min_element(data.begin(), data.end());
  max = *std::max_element(data.begin(), data.end());
  avg = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
  double sum_sq = 0.0;
  for (double v : data) sum_sq += (v - avg) * (v - avg);
  stddev = std::sqrt(sum_sq / data.size());
}

int main() {
  std::ofstream csv("C_kernel_perf_in_C.csv");
  int trial_num = 50;

  csv << "slice_length,log2_msm_length,msm_length,buf_extend_ratio,signed_"
         "bucket,min,max,avg(us),stddev\n";

  std::vector<uint32_t> slice_lengths = {6, 8, 10, 12, 14};
  std::vector<uint32_t> log2_msm_lengths = {10, 12, 14,
                                            16, 18, 20};  // log2(msm_length)
  std::vector<double> buf_extend_ratios = {1.1};
  std::vector<bool> signed_buckets = {false, true};

  for (uint32_t slice_length : slice_lengths) {
    for (uint32_t log2_msm_length : log2_msm_lengths) {
      uint32_t msm_length = 1U << log2_msm_length;
      for (double buf_extend_ratio : buf_extend_ratios) {
        for (bool signed_bucket : signed_buckets) {
          if (slice_length >= log2_msm_length) {
            // std::cerr << "Error: slice_length must be less than
            // log2_msm_length." << std::endl;
            continue;
          }
          std::cout << "Testing slice_length: " << slice_length
                    << ", log2_msm_length: " << log2_msm_length
                    << ", buf_extend_ratio: " << buf_extend_ratio
                    << ", signed_bucket: " << (signed_bucket ? "true" : "false")
                    << std::endl;
          // Run the test multiple times for statistics
          std::vector<double> results;
          for (int trial = 0; trial < trial_num; ++trial) {
            double perf = run_performance_test(slice_length, msm_length,
                                               buf_extend_ratio, signed_bucket);
            results.push_back(perf);
          }
          double min, max, avg, stddev;
          compute_stats(results, min, max, avg, stddev);

          csv << slice_length << "," << log2_msm_length << "," << msm_length
              << "," << buf_extend_ratio << "," << signed_bucket << "," << min
              << "," << max << "," << avg << "," << stddev << "\n";
        }
      }
    }
  }

  csv.close();
  std::cout << "Performance results written to performance_results.csv\n";
  return 0;
}

namespace ffi = xla::ffi;
ffi::Error DistributeImpl(uint32_t window_num, uint32_t regular_bucket_num,
                          uint32_t special_bucket_num, uint32_t msm_length,
                          uint32_t fixed_regular_padding_size,
                          uint32_t fixed_special_padding_size,
                          ffi::Buffer<ffi::S32> slices_list,
                          ffi::Buffer<ffi::U32> points_list,
                          ffi::Buffer<ffi::U32> zero,
                          ffi::ResultBuffer<ffi::U32> regular_buckets,
                          ffi::ResultBuffer<ffi::U32> special_buckets,
                          ffi::ResultBuffer<ffi::U32> metadata) {
  // Create a Distributor object
  Distributor distributor(window_num, regular_bucket_num, special_bucket_num,
                          msm_length);
  distributor.set_fixed_padding_size(fixed_regular_padding_size,
                                     fixed_special_padding_size);
  distributor.set_slices_list(slices_list.typed_data());
  distributor.set_points_list(points_list.typed_data());
  distributor.set_zeros(zero.typed_data());

  distributor.distribute();

  distributor.get_merged_regular_buckets(regular_buckets->typed_data());
  distributor.get_merged_special_buckets(special_buckets->typed_data());
  distributor.get_merged_metadata(metadata->typed_data());
  // Return success
  return xla::ffi::Error::Success();
}

ffi::Error DistributeBufImpl(uint32_t window_num, uint32_t regular_bucket_num,
                             uint32_t special_bucket_num, uint32_t msm_length,
                             uint32_t fixed_regular_padding_size,
                             uint32_t fixed_special_padding_size,
                             ffi::Buffer<ffi::S32> slices_list,
                             ffi::Buffer<ffi::U32> points_list,
                             ffi::Buffer<ffi::U32> zero,
                             ffi::ResultBuffer<ffi::U32> regular_buckets,
                             ffi::ResultBuffer<ffi::U32> special_buckets,
                             ffi::ResultBuffer<ffi::U32> metadata) {
  // Create a Distributor object
  Distributor distributor(window_num, regular_bucket_num, special_bucket_num,
                          msm_length);
  distributor.set_fixed_padding_size(fixed_regular_padding_size,
                                     fixed_special_padding_size);
  distributor.set_slices_list(slices_list.typed_data());
  distributor.set_points_list(points_list.typed_data());
  distributor.set_zeros(zero.typed_data());
  distributor.set_output_buffers(regular_buckets->typed_data(),
                                 special_buckets->typed_data());

  distributor.distribute_to_buffer_parallel();

  // Return success
  return ffi::Error::Success();
}

ffi::Error DistributeBufSignedImpl(
    uint32_t window_num, uint32_t regular_bucket_num,
    uint32_t special_bucket_num, uint32_t msm_length,
    uint32_t fixed_regular_padding_size, uint32_t fixed_special_padding_size,
    ffi::Buffer<ffi::S32> slices_list, ffi::Buffer<ffi::U32> points_list,
    ffi::Buffer<ffi::U32> neg_points_list, ffi::Buffer<ffi::U32> zero,
    ffi::ResultBuffer<ffi::U32> regular_buckets,
    ffi::ResultBuffer<ffi::U32> special_buckets,
    ffi::ResultBuffer<ffi::U32> metadata) {
  // Create a Distributor object
  Distributor distributor(window_num, regular_bucket_num, special_bucket_num,
                          msm_length);
  distributor.set_fixed_padding_size(fixed_regular_padding_size,
                                     fixed_special_padding_size);
  distributor.set_slices_list(slices_list.typed_data());
  distributor.set_points_list(points_list.typed_data());
  distributor.set_neg_points_list(neg_points_list.typed_data());
  distributor.set_zeros(zero.typed_data());
  distributor.set_output_buffers(regular_buckets->typed_data(),
                                 special_buckets->typed_data());

  distributor.distribute_to_buffer_signed_parallel();

  // Return success
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Distribute, DistributeImpl,
    ffi::Ffi::Bind()
        .Attr<uint32_t>("window_num")
        .Attr<uint32_t>("regular_bucket_num")
        .Attr<uint32_t>("special_bucket_num")
        .Attr<uint32_t>("msm_length")
        .Attr<uint32_t>("fixed_regular_padding_size")
        .Attr<uint32_t>("fixed_special_padding_size")
        .Arg<ffi::Buffer<ffi::S32>>()  // slices_list
        .Arg<ffi::Buffer<ffi::U32>>()  // points_list
        .Arg<ffi::Buffer<ffi::U32>>()  // zero
        .Ret<ffi::Buffer<ffi::U32>>()  // regular_buckets
        .Ret<ffi::Buffer<ffi::U32>>()  // special_buckets
        .Ret<ffi::Buffer<ffi::U32>>()  // metadata
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DistributeBuf, DistributeBufImpl,
    ffi::Ffi::Bind()
        .Attr<uint32_t>("window_num")
        .Attr<uint32_t>("regular_bucket_num")
        .Attr<uint32_t>("special_bucket_num")
        .Attr<uint32_t>("msm_length")
        .Attr<uint32_t>("fixed_regular_padding_size")
        .Attr<uint32_t>("fixed_special_padding_size")
        .Arg<ffi::Buffer<ffi::S32>>()  // slices_list
        .Arg<ffi::Buffer<ffi::U32>>()  // points_list
        .Arg<ffi::Buffer<ffi::U32>>()  // zero
        .Ret<ffi::Buffer<ffi::U32>>()  // regular_buckets
        .Ret<ffi::Buffer<ffi::U32>>()  // special_buckets
        .Ret<ffi::Buffer<ffi::U32>>()  // metadata
);

ffi::Error DistributeBufZeroImpl(
    uint32_t window_num, uint32_t regular_bucket_num,
    uint32_t special_bucket_num, uint32_t msm_length,
    uint32_t fixed_regular_padding_size, uint32_t fixed_special_padding_size,
    ffi::Buffer<ffi::S32> slices_list, ffi::Buffer<ffi::U32> points_list,
    ffi::Buffer<ffi::U32> zero, ffi::ResultBuffer<ffi::U32> regular_buckets,
    ffi::ResultBuffer<ffi::U32> special_buckets,
    ffi::ResultBuffer<ffi::U32> metadata) {
  Distributor distributor(window_num, regular_bucket_num, special_bucket_num,
                          msm_length);
  distributor.set_fixed_padding_size(fixed_regular_padding_size,
                                     fixed_special_padding_size);
  distributor.set_slices_list(slices_list.typed_data());
  distributor.set_points_list(points_list.typed_data());
  distributor.set_zeros(zero.typed_data());
  distributor.set_output_buffers(regular_buckets->typed_data(),
                                 special_buckets->typed_data());

  distributor.distribute_to_buffer_zero_indexed_parallel();

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DistributeBufZero, DistributeBufZeroImpl,
    ffi::Ffi::Bind()
        .Attr<uint32_t>("window_num")
        .Attr<uint32_t>("regular_bucket_num")
        .Attr<uint32_t>("special_bucket_num")
        .Attr<uint32_t>("msm_length")
        .Attr<uint32_t>("fixed_regular_padding_size")
        .Attr<uint32_t>("fixed_special_padding_size")
        .Arg<ffi::Buffer<ffi::S32>>()  // slices_list
        .Arg<ffi::Buffer<ffi::U32>>()  // points_list
        .Arg<ffi::Buffer<ffi::U32>>()  // zero
        .Ret<ffi::Buffer<ffi::U32>>()  // regular_buckets
        .Ret<ffi::Buffer<ffi::U32>>()  // special_buckets
        .Ret<ffi::Buffer<ffi::U32>>()  // metadata
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DistributeBufSigned, DistributeBufSignedImpl,
    ffi::Ffi::Bind()
        .Attr<uint32_t>("window_num")
        .Attr<uint32_t>("regular_bucket_num")
        .Attr<uint32_t>("special_bucket_num")
        .Attr<uint32_t>("msm_length")
        .Attr<uint32_t>("fixed_regular_padding_size")
        .Attr<uint32_t>("fixed_special_padding_size")
        .Arg<ffi::Buffer<ffi::S32>>()  // slices_list
        .Arg<ffi::Buffer<ffi::U32>>()  // points_list
        .Arg<ffi::Buffer<ffi::U32>>()  // neg_points_list
        .Arg<ffi::Buffer<ffi::U32>>()  // zero
        .Ret<ffi::Buffer<ffi::U32>>()  // regular_buckets
        .Ret<ffi::Buffer<ffi::U32>>()  // special_buckets
        .Ret<ffi::Buffer<ffi::U32>>()  // metadata
);
