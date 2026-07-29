/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifdef ENABLE_CPU_TRACE_INTRINSIC
#else
#include "Debug/Debug.h"
#include <type_traits>

[aicore] __attribute__((always_inline)) float bf162f32(bfloat16_t v) {
  // the hardware does not support static_cast bfloat16_t -> float
  // and in this case left-shift is well-defined for bfloat16_t -> float
  uint16_t ui16 = *reinterpret_cast<uint16_t *>(&v);
  uint32_t ui32 = static_cast<uint32_t>(ui16);
  ui32 <<= 16;
  return *reinterpret_cast<float *>(&ui32);
}

template <typename T>
[aicore] __attribute__((always_inline)) float cast2f32(T v) {
  if constexpr (std::is_same<T, bfloat16_t>::value) {
    return bf162f32(v);
  }
  if constexpr (std::is_same<T, half>::value) {
    return static_cast<float>(v);
  }
  if constexpr (std::is_same<T, float>::value) {
    return v;
  }
}

template <typename T>
[aicore] __attribute__((always_inline)) void
print_core_prefix_fmt(char *prefix, const int64_t len, const int8_t hex,
                      const __gm__ char **nlast_elem_fmt,
                      const __gm__ char **last_elem_fmt) {
  const char *prefix_ptr = static_cast<const char *>(prefix);
  for (int64_t i = 0; i < len; i++) {
    cce::printf("%c", prefix_ptr[i]);
  }
  cce::printf("\n");

  if constexpr (std::is_same<T, int8_t>::value ||
                std::is_same<T, int16_t>::value ||
                std::is_same<T, int32_t>::value) {
    if (hex) {
      *nlast_elem_fmt = "%X,";
      *last_elem_fmt = "%X";
    } else {
      *nlast_elem_fmt = "%d,";
      *last_elem_fmt = "%d";
    }
  } else if constexpr (std::is_same<T, int64_t>::value) {
    if (hex) {
      *nlast_elem_fmt = "%llX,";
      *last_elem_fmt = "%llX";
    } else {
      *nlast_elem_fmt = "%lld,";
      *last_elem_fmt = "%lld";
    }
  } else if constexpr (std::is_same<T, uint8_t>::value ||
                       std::is_same<T, uint16_t>::value ||
                       std::is_same<T, uint32_t>::value) {
    if (hex) {
      *nlast_elem_fmt = "%X,";
      *last_elem_fmt = "%X";
    } else {
      *nlast_elem_fmt = "%u,";
      *last_elem_fmt = "%u";
    }
  } else if constexpr (std::is_same<T, uint64_t>::value) {
    if (hex) {
      *nlast_elem_fmt = "%llX,";
      *last_elem_fmt = "%llX";
    } else {
      *nlast_elem_fmt = "%llu,";
      *last_elem_fmt = "%llu";
    }
  } else if constexpr (std::is_same<T, half>::value ||
                       std::is_same<T, bfloat16_t>::value ||
                       std::is_same<T, float>::value) {
    if (hex) {
      *nlast_elem_fmt = "%A,";
      *last_elem_fmt = "%A";
    } else {
      *nlast_elem_fmt = "%f,";
      *last_elem_fmt = "%f";
    }
  } else if constexpr (std::is_same<T, bool>::value) {
    *nlast_elem_fmt = "%c,";
    *last_elem_fmt = "%c";
  }
}

[aicore] __attribute__((always_inline)) __ubuf__ uint8_t *
cast_bool_to_uint8_t(__ubuf__ bool *ptr) {
  return reinterpret_cast<__ubuf__ uint8_t*>(ptr);
}

[aicore] __attribute__((always_inline)) __gm__ uint8_t *
cast_bool_to_uint8_t(__gm__ bool *ptr) {
  return reinterpret_cast<__gm__ uint8_t*>(ptr);
}

template <typename T>
[aicore] __attribute__((always_inline)) void
print_scalar_core(char *prefix, const int64_t len, T arg, const int8_t hex) {
  pipe_barrier(PIPE_ALL);
  // the fmt of nlast elem followed by `,`
  const __gm__ char *nlast_elem_fmt;
  // the fmt of last elem followed by `\n`
  const __gm__ char *last_elem_fmt;

  print_core_prefix_fmt<T>(prefix, len, hex, &nlast_elem_fmt, &last_elem_fmt);

  if constexpr (std::is_same<T, bool>::value) {
    char val = arg ? 'T' : 'F';
    cce::printf(last_elem_fmt, val);
  } else if constexpr (std::is_same<T, half>::value ||
                       std::is_same<T, bfloat16_t>::value ||
                       std::is_same<T, float>::value) {
    cce::printf(last_elem_fmt, cast2f32(arg));
  } else {
    cce::printf(last_elem_fmt, arg);
  }
  cce::printf("\n");
}

namespace { // anonymous namespace to keep helper functions private in TU
template <std::size_t RANK>
[aicore] __attribute__((always_inline)) void
cce_printf_valid_tensor_type(int* cur_iters) {
  cce::printf("(");
  for (int i = 0; i < RANK - 2; ++i) {
    cce::printf("%d, ", cur_iters[static_cast<std::size_t>(i)]);
   }
  cce::printf(":, :):\n");
}

// сompile‑time recursion over dimensions. the loop nest is fully unrolled
// for each 'CUR_DIM' via if constexpr, which results in zero runtime recursion overhead.
template<typename T, typename MEM_T, std::size_t CUR_DIM, std::size_t RANK>
[aicore] __attribute__((always_inline)) void
print_nd_core_recursive(const int64_t len, memref_t<MEM_T, RANK> *arg,
                        const int8_t hex, int64_t base,
                        int* cur_iters,
                        const __gm__ char** last_elem_fmt,
                        const __gm__ char** nlast_elem_fmt,
                        MEM_T *arg_ptr) {
  // if we are on the last recursion step, then CUR_DIM == RANK.
  // but we can't arg->sizes[RANK] if size(arg->sizes) == RANK
  // so we need clamped CUR_DIM
  auto clamped_cur_dim = (CUR_DIM == RANK) ? CUR_DIM - 1 : CUR_DIM;
  auto size_i = arg->sizes[clamped_cur_dim];
  auto size_i_dec = arg->sizes[(clamped_cur_dim - 1) >= 0 ? clamped_cur_dim - 1 : 0];
  auto stride_i = arg->strides[clamped_cur_dim];
  auto stride_i_dec = arg->strides[(clamped_cur_dim - 1) >= 0 ? clamped_cur_dim - 1 : 0];
  float val;
  if constexpr (RANK > 2 && CUR_DIM == RANK - 2) {
    cce_printf_valid_tensor_type<RANK>(cur_iters);
  }
  if constexpr (CUR_DIM == RANK - 2 || CUR_DIM == RANK - 1 || RANK == 1) {
    cce::printf("[");
  }
  if constexpr (CUR_DIM == RANK - 1) {
    if constexpr (std::is_same_v<T, half> ||
                  std::is_same_v<T, bfloat16_t> ||
                  std::is_same_v<T, float>) {
      for (int iter = 0; iter < size_i - 1; ++iter) {
        val = cast2f32(arg_ptr[base + iter * stride_i]);
        cce::printf(*nlast_elem_fmt, val);
      }
      int64_t closing_offset = base + (size_i - 1) * stride_i;
      val = cast2f32(arg_ptr[closing_offset]);
      cce::printf(*last_elem_fmt, val);
      cce::printf(cur_iters[CUR_DIM-1] < size_i_dec - 1 ? "],\n" : "]");
    } else if constexpr (std::is_same_v<T, bool>) {
      auto *byte_ptr = cast_bool_to_uint8_t(arg_ptr);
      for (int iter = 0; iter < size_i; ++iter) {
        auto offset = base + iter * stride_i;
        int64_t byte_offset = offset / 8;
        int64_t bit_offset = offset % 8;
        char val =  ((byte_ptr[byte_offset]) >> bit_offset) & 1 ? 'T' : 'F';
                  cce::printf(iter < size_i - 1 ? *nlast_elem_fmt : *last_elem_fmt,
                              val);
      }
      cce::printf(cur_iters[CUR_DIM-1] < size_i_dec - 1 ? "],\n" : "]");
    } else {
      for (int iter = 0; iter < size_i - 1; ++iter) {
        cce::printf(*nlast_elem_fmt, arg_ptr[base + iter * stride_i]);
      }
      int64_t closing_offset = base + (size_i - 1) * stride_i;
      cce::printf(*last_elem_fmt, arg_ptr[closing_offset]);
      cce::printf(cur_iters[CUR_DIM-1] < size_i_dec - 1 ? "],\n" : "]");
    }
  } else {
    for (int iter = 0; iter < size_i; ++iter) {
    // we count base until we finally get to the last recursive layer and the next
    // call is going to print the contents of array, iterating by base + i * stride_(n-1)
    auto new_base = base + iter * stride_i;
    cur_iters[static_cast<std::size_t>(CUR_DIM)] = iter;
    print_nd_core_recursive<T, MEM_T, CUR_DIM + 1, RANK>(len, arg, hex, new_base, cur_iters,
                                                                                  last_elem_fmt,
                                                                                  nlast_elem_fmt,
                                                                                  arg_ptr);
    }
  }
  if constexpr (CUR_DIM == RANK - 2 || RANK == 1) {
    cce::printf("]\n");
  }
}
} // end of anonymous namespace

template <typename T, typename MEM_T, std::size_t RANK>
[aicore] __attribute__((always_inline)) void
print_nd_core(char *prefix, const int64_t len, memref_t<MEM_T, RANK> *arg,
               const int8_t hex) {
  pipe_barrier(PIPE_ALL);
  const __gm__ char *nlast_elem_fmt;
  const __gm__ char *last_elem_fmt;
  print_core_prefix_fmt<T>(prefix, len, hex, &nlast_elem_fmt, &last_elem_fmt);
  auto arg_ptr = arg->aligned + arg->offset;
  int curIters[RANK] = {};
  print_nd_core_recursive<T, MEM_T, 0, RANK>(len, arg, hex, 0,        // base per depth param
                                                            curIters, // C-array of cycle iters per recursion depth
                                                            &last_elem_fmt,
                                                            &nlast_elem_fmt,
                                                            arg_ptr);
}

template <typename T>
[aicore] __attribute__((always_inline)) void
print_assert_msg(const T *msg, const int64_t len) {
  const char *msg_ptr = static_cast<const char *>(msg);
  for (int64_t i = 0; i < len; i++) {
    cce::printf("%c", msg_ptr[i]);
  }
}

template <typename T>
[aicore] __attribute__((always_inline)) void
npuir_cce_assert(bool cond, const T *msg, const int64_t len) {
  if (!cond) {
#ifdef __CCE_AICORE_ENABLE_MIX__
#ifdef __CCE_ENABLE_PRINT_AICORE_CUBE__
    cce::printf("*** Assertion Failure (Cube, Block ID = %d): ",
                get_block_idx());
    print_assert_msg(msg, len);
#endif
#ifdef __CCE_ENABLE_PRINT_AICORE_VEC__
    cce::printf(
        "*** Assertion Failure (Vec, Block ID = %d, SubBlock ID = %d): ",
        get_block_idx(), get_subblockid());
    print_assert_msg(msg, len);
#endif
#else
#ifdef __CCE_ENABLE_PRINT_AICORE_CUBE__
    cce::printf("*** Assertion Failure (Cube, Block ID = %d): ",
                get_block_idx());
    print_assert_msg(msg, len);
#endif
#ifdef __CCE_ENABLE_PRINT_AICORE_VEC__
    cce::printf("*** Assertion Failure (Vec, Block ID = %d): ",
                get_block_idx());
    print_assert_msg(msg, len);
#endif
#endif
    trap();
  }
}

[aicore] __attribute__((always_inline)) void
assert_scalar_core(char *prefix, const int64_t len, bool arg) {
  pipe_barrier(PIPE_ALL);
  npuir_cce_assert(arg, prefix, len);
}

namespace { // anonymous namespace to keep helper functions private in TU
template<typename MEM_T, std::size_t CUR_DIM, std::size_t RANK>
[aicore] __attribute__((always_inline)) void
assert_nd_core_recursive(char *prefix, const int64_t len, memref_t<MEM_T, RANK> *arg, int64_t base,
                                                                                      MEM_T *arg_ptr) {
  auto clamped_cur_dim = (CUR_DIM == RANK) ? CUR_DIM - 1: CUR_DIM;
  auto size_i = arg->sizes[clamped_cur_dim];
  auto stride_i = arg->strides[clamped_cur_dim];

  if constexpr (CUR_DIM == RANK - 1) {
    for (int iter = 0; iter < size_i; ++iter) {
      if (!arg_ptr[base + iter * stride_i]) {
         npuir_cce_assert(false, prefix, len);
       }
     }
       } else {
    for (int iter = 0; iter < size_i; ++iter) {
      auto new_base = base + iter * stride_i;
      assert_nd_core_recursive<MEM_T, CUR_DIM + 1, RANK>(prefix, len, arg, new_base,
                                                                           arg_ptr);
     }
   }
 }
} // end of anonymous namespace

template<typename MEM_T, std::size_t RANK>
[aicore] __attribute__((always_inline)) void
assert_nd_core(char *prefix, const int64_t len, memref_t<MEM_T, RANK> *arg) {
   pipe_barrier(PIPE_ALL);
   auto arg_ptr = arg->aligned + arg->offset;
   assert_nd_core_recursive<MEM_T, 0, RANK>(prefix, len, arg, 0, // base param
                                                         arg_ptr);
 }


extern "C" {
// register __gm__ versions for both cube core and vector core

REGISTER_PRINT_SCALAR(int8_t, gm)
REGISTER_PRINT_SCALAR(uint8_t, gm)
REGISTER_PRINT_SCALAR(int16_t, gm)
REGISTER_PRINT_SCALAR(uint16_t, gm)
REGISTER_PRINT_SCALAR(int32_t, gm)
REGISTER_PRINT_SCALAR(uint32_t, gm)
REGISTER_PRINT_SCALAR(int64_t, gm)
REGISTER_PRINT_SCALAR(half, gm)
REGISTER_PRINT_SCALAR(bfloat16_t, gm)
REGISTER_PRINT_SCALAR(float, gm)
REGISTER_PRINT_SCALAR(bool, gm)
REGISTER_PRINT_1TO8D_TENSOR(int8_t, gm)
REGISTER_PRINT_1TO8D_TENSOR(uint8_t, gm)
REGISTER_PRINT_1TO8D_TENSOR(int16_t, gm)
REGISTER_PRINT_1TO8D_TENSOR(uint16_t, gm)
REGISTER_PRINT_1TO8D_TENSOR(int32_t, gm)
REGISTER_PRINT_1TO8D_TENSOR(uint32_t, gm)
REGISTER_PRINT_1TO8D_TENSOR(int64_t, gm)
REGISTER_PRINT_1TO8D_TENSOR(half, gm)
REGISTER_PRINT_1TO8D_TENSOR(bfloat16_t, gm)
REGISTER_PRINT_1TO8D_TENSOR(float, gm)
REGISTER_PRINT_1TO8D_TENSOR(bool, gm)
REGISTER_ASSERT_SCALAR(gm)
REGISTER_ASSERT_1TO4D_TENSOR(gm)

// register __ubuf__ versions for vector core
// Note: bisheng uses the following macro for both print and assert

#ifdef __CCE_ENABLE_PRINT_AICORE_VEC__
REGISTER_PRINT_SCALAR(int8_t, ubuf)
REGISTER_PRINT_SCALAR(uint8_t, ubuf)
REGISTER_PRINT_SCALAR(int16_t, ubuf)
REGISTER_PRINT_SCALAR(uint16_t, ubuf)
REGISTER_PRINT_SCALAR(int32_t, ubuf)
REGISTER_PRINT_SCALAR(uint32_t, ubuf)
REGISTER_PRINT_SCALAR(int64_t, ubuf)
REGISTER_PRINT_SCALAR(half, ubuf)
REGISTER_PRINT_SCALAR(bfloat16_t, ubuf)
REGISTER_PRINT_SCALAR(float, ubuf)
REGISTER_PRINT_SCALAR(bool, ubuf)
REGISTER_PRINT_1TO8D_TENSOR(int8_t, ubuf)
REGISTER_PRINT_1TO8D_TENSOR(uint8_t, ubuf)
REGISTER_PRINT_1TO8D_TENSOR(int16_t, ubuf)
REGISTER_PRINT_1TO8D_TENSOR(uint16_t, ubuf)
REGISTER_PRINT_1TO8D_TENSOR(int32_t, ubuf)
REGISTER_PRINT_1TO8D_TENSOR(uint32_t, ubuf)
REGISTER_PRINT_1TO8D_TENSOR(int64_t, ubuf)
REGISTER_PRINT_1TO8D_TENSOR(half, ubuf)
REGISTER_PRINT_1TO8D_TENSOR(bfloat16_t, ubuf)
REGISTER_PRINT_1TO8D_TENSOR(float, ubuf)
REGISTER_PRINT_1TO8D_TENSOR(bool, ubuf)
REGISTER_ASSERT_SCALAR(ubuf)
REGISTER_ASSERT_1TO4D_TENSOR(ubuf)
#endif

[aicore] __attribute__((always_inline)) void
_mlir_ciface_init_debug(__gm__ cce::internal::DebugTunnelData *DTData) {
  cce::internal::DebugTunnel::OnKernelInitialize(DTData);
}
[aicore] __attribute__((always_inline)) void
_mlir_ciface_finish_debug(__gm__ cce::internal::DebugTunnelData *DTData) {
  cce::internal::DebugTunnel::OnKernelFinish(DTData);
}
}
#endif
