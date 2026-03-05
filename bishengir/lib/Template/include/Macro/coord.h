/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

template <uint32_t kRank, typename Index = uint32_t> struct Coord {
public:
  //
  // Methods
  //

  /// Default ctor initializes uniformly
  __aicore__
      __attribute__((always_inline)) explicit Coord(Index value = Index(0)) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] = value;
    }
  }

  /// Constructs from an array of integers
  __aicore__ __attribute__((always_inline)) Coord(Index const (&_idx)[kRank]) {
    for (int i = 0; i < kRank; ++i) {
      idx[i] = _idx[i];
    }
  }

  /// Returns the index of the dimension with least value
  __aicore__ __attribute__((always_inline)) uint32_t min_dim_index() const {
    uint32_t i = 0;
    for (uint32_t j = 1; j < kRank; ++j) {
      if (idx[j] < idx[i]) {
        i = j;
      }
    }
    return i;
  }

  /// Returns the index of the dimension with greatest value
  __aicore__ __attribute__((always_inline)) uint32_t max_dim_index() const {
    uint32_t i = 0;
    for (uint32_t j = 1; j < kRank; ++j) {
      if (idx[j] > idx[i]) {
        i = j;
      }
    }
    return i;
  }

  /// Returns true if Coord is non-zero.
  __aicore__ __attribute__((always_inline)) explicit operator bool() const {
    for (int i = 0; i < kRank; ++i) {
      if (idx[i]) {
        return true;
      }
    }
    return false;
  }

  /// Member access operator
  __aicore__ __attribute__((always_inline)) Index &operator[](int dim) {
    return idx[dim];
  }

  /// Member access operator
  __aicore__ __attribute__((always_inline)) Index const &
  operator[](int dim) const {
    return idx[dim];
  }

  /// Access via index; may limit unrolling potential
  __aicore__ __attribute__((always_inline)) Index &at(int dim) {
    return idx[dim];
  }

private:
  //
  // Data members
  //

  /// Indices
  Index idx[kRank];
};
