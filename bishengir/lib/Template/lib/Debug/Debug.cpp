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
#include "Debug/DebugUtils.h"
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
}

template <typename T, typename MEM_T>
[aicore] __attribute__((always_inline)) void
print_1d_core(char *prefix, const int64_t len, memref_t<MEM_T, 1> *arg,
              const int8_t hex) {
  pipe_barrier(PIPE_ALL);
  // the fmt of nlast elem followed by `,`
  const __gm__ char *nlast_elem_fmt;
  // the fmt of last elem followed by `\n`
  const __gm__ char *last_elem_fmt;

  print_core_prefix_fmt<T>(prefix, len, hex, &nlast_elem_fmt, &last_elem_fmt);
  cce::printf("[");

  auto arg_ptr = arg->aligned + arg->offset;
  const int64_t size0 = arg->sizes[0];
  const int64_t stride0 = arg->strides[0];
  if constexpr (std::is_same<T, half>::value ||
                std::is_same<T, bfloat16_t>::value ||
                std::is_same<T, float>::value) {
    float val;
    for (int64_t i = 0; i < size0 - 1; i++) {
      val = cast2f32(arg_ptr[i * stride0]);
      cce::printf(nlast_elem_fmt, val);
    }
    val = cast2f32(arg_ptr[(size0 - 1) * stride0]);
    cce::printf(last_elem_fmt, val);
  } else if constexpr (std::is_same<T, bool>::value) {
    auto *byte_ptr = cast_bool_to_uint8_t(arg_ptr);
    for (int64_t i = 0; i < size0; i++) {
      int64_t byte_offset = i / 8;
      int64_t bit_offset = i % 8;
      char val = ((byte_ptr[byte_offset]) >> bit_offset) & 1 ? 'T' : 'F';
      cce::printf(i < size0 - 1 ? nlast_elem_fmt : last_elem_fmt, val);
    }
  } else {
    for (int64_t i = 0; i < size0 - 1; i++) {
      cce::printf(nlast_elem_fmt, arg_ptr[i * stride0]);
    }
    cce::printf(last_elem_fmt, arg_ptr[(size0 - 1) * stride0]);
  }
  cce::printf("]\n");
}

template <typename T, typename MEM_T>
[aicore] __attribute__((always_inline)) void
print_2d_core(char *prefix, const int64_t len, memref_t<MEM_T, 2> *arg,
              const int8_t hex) {
  pipe_barrier(PIPE_ALL);
  const __gm__ char *nlast_elem_fmt;
  const __gm__ char *last_elem_fmt;
  print_core_prefix_fmt<T>(prefix, len, hex, &nlast_elem_fmt, &last_elem_fmt);
  cce::printf("[");

  auto arg_ptr = arg->aligned + arg->offset;
  const int64_t size0 = arg->sizes[0];
  const int64_t stride0 = arg->strides[0];
  const int64_t size1 = arg->sizes[1];
  const int64_t stride1 = arg->strides[1];
  if constexpr (std::is_same<T, half>::value ||
                std::is_same<T, bfloat16_t>::value ||
                std::is_same<T, float>::value) {
    float val;
    for (int64_t i0 = 0; i0 < size0; i0++) {
      cce::printf("[");
      int64_t base = i0 * stride0;
      for (int64_t i1 = 0; i1 < size1 - 1; i1++) {
        int64_t i = base + i1 * stride1;
        val = cast2f32(arg_ptr[i]);
        cce::printf(nlast_elem_fmt, val);
      }
      int64_t i = base + (size1 - 1) * stride1;
      val = cast2f32(arg_ptr[i]);
      cce::printf(last_elem_fmt, val);
      cce::printf(i0 < size0 - 1 ? "],\n" : "]");
    }
  } else if constexpr (std::is_same<T, bool>::value) {
    auto *byte_ptr = cast_bool_to_uint8_t(arg_ptr);
    for (int64_t i0 = 0; i0 < size0; i0++) {
      cce::printf("[");
      for (int64_t i1 = 0; i1 < size1; i1++) {
        int64_t i = i0 * stride0 + i1 * stride1;
        int64_t byte_offset = i / 8;
        int64_t bit_offset = i % 8;
        char val = ((byte_ptr[byte_offset]) >> bit_offset) & 1 ? 'T' : 'F';
        cce::printf(i1 < size1 - 1 ? nlast_elem_fmt : last_elem_fmt, val);
      }
      cce::printf(i0 < size0 - 1 ? "],\n" : "]");
    }
  } else {
    for (int64_t i0 = 0; i0 < size0; i0++) {
      cce::printf("[");
      int64_t base = i0 * stride0;
      for (int64_t i1 = 0; i1 < size1 - 1; i1++) {
        int64_t i = base + i1 * stride1;
        cce::printf(nlast_elem_fmt, arg_ptr[i]);
      }
      int64_t i = base + (size1 - 1) * stride1;
      cce::printf(last_elem_fmt, arg_ptr[i]);
      cce::printf(i0 < size0 - 1 ? "],\n" : "]");
    }
  }
  cce::printf("]\n");
}

template <typename T, typename MEM_T>
[aicore] __attribute__((always_inline)) void
print_3d_core(char *prefix, const int64_t len, memref_t<MEM_T, 3> *arg,
              const int8_t hex) {
  pipe_barrier(PIPE_ALL);
  const __gm__ char *nlast_elem_fmt;
  const __gm__ char *last_elem_fmt;
  print_core_prefix_fmt<T>(prefix, len, hex, &nlast_elem_fmt, &last_elem_fmt);

  auto arg_ptr = arg->aligned + arg->offset;
  const int64_t size0 = arg->sizes[0];
  const int64_t stride0 = arg->strides[0];
  const int64_t size1 = arg->sizes[1];
  const int64_t stride1 = arg->strides[1];
  const int64_t size2 = arg->sizes[2];
  const int64_t stride2 = arg->strides[2];
  if constexpr (std::is_same<T, half>::value ||
                std::is_same<T, bfloat16_t>::value ||
                std::is_same<T, float>::value) {
    float val;
    for (int64_t i0 = 0; i0 < size0; i0++) {
      cce::printf("(%d, :, :):\n", i0);
      cce::printf("[");
      for (int64_t i1 = 0; i1 < size1; i1++) {
        cce::printf("[");
        int64_t base = i0 * stride0 + i1 * stride1;
        for (int64_t i2 = 0; i2 < size2 - 1; i2++) {
          int64_t i = base + i2 * stride2;
          val = cast2f32(arg_ptr[i]);
          cce::printf(nlast_elem_fmt, val);
        }
        int64_t i = base + (size2 - 1) * stride2;
        val = cast2f32(arg_ptr[i]);
        cce::printf(last_elem_fmt, val);
        cce::printf(i1 < size1 - 1 ? "],\n" : "]");
      }
      cce::printf("]\n");
    }
  } else if constexpr (std::is_same<T, bool>::value) {
    auto *byte_ptr = cast_bool_to_uint8_t(arg_ptr);
    for (int64_t i0 = 0; i0 < size0; i0++) {
      cce::printf("(%d, :, :):\n", i0);
      cce::printf("[");
      for (int64_t i1 = 0; i1 < size1; i1++) {
        cce::printf("[");
        for (int64_t i2 = 0; i2 < size2; i2++) {
          int64_t i = i0 * stride0 + i1 * stride1 + i2 * stride2;
          int64_t byte_offset = i / 8;
          int64_t bit_offset = i % 8;
          char val = ((byte_ptr[byte_offset]) >> bit_offset) & 1 ? 'T' : 'F';
          cce::printf(i2 < size2 - 1 ? nlast_elem_fmt : last_elem_fmt, val);
        }
        cce::printf(i1 < size1 - 1 ? "],\n" : "]");
      }
      cce::printf("]\n");
    }
  } else {
    for (int64_t i0 = 0; i0 < size0; i0++) {
      cce::printf("(%d, :, :):\n", i0);
      cce::printf("[");
      for (int64_t i1 = 0; i1 < size1; i1++) {
        cce::printf("[");
        int64_t base = i0 * stride0 + i1 * stride1;
        for (int64_t i2 = 0; i2 < size2 - 1; i2++) {
          int64_t i = base + i2 * stride2;
          cce::printf(nlast_elem_fmt, arg_ptr[i]);
        }
        int64_t i = base + (size2 - 1) * stride2;
        cce::printf(last_elem_fmt, arg_ptr[i]);
        cce::printf(i1 < size1 - 1 ? "],\n" : "]");
      }
      cce::printf("]\n");
    }
  }
}

template <typename T, typename MEM_T>
[aicore] __attribute__((always_inline)) void
print_4d_core(char *prefix, const int64_t len, memref_t<MEM_T, 4> *arg,
              const int8_t hex) {
  pipe_barrier(PIPE_ALL);

  const __gm__ char *nlast_elem_fmt;
  const __gm__ char *last_elem_fmt;
  print_core_prefix_fmt<T>(prefix, len, hex, &nlast_elem_fmt, &last_elem_fmt);

  auto arg_ptr = arg->aligned + arg->offset;
  const int64_t size0 = arg->sizes[0];
  const int64_t stride0 = arg->strides[0];
  const int64_t size1 = arg->sizes[1];
  const int64_t stride1 = arg->strides[1];
  const int64_t size2 = arg->sizes[2];
  const int64_t stride2 = arg->strides[2];
  const int64_t size3 = arg->sizes[3];
  const int64_t stride3 = arg->strides[3];
  if constexpr (std::is_same<T, half>::value ||
                std::is_same<T, bfloat16_t>::value ||
                std::is_same<T, float>::value) {
    float val;
    for (int64_t i0 = 0; i0 < size0; i0++) {
      for (int64_t i1 = 0; i1 < size1; i1++) {
        cce::printf("(%d, %d, :, :):\n", i0, i1);
        cce::printf("[");
        for (int64_t i2 = 0; i2 < size2; i2++) {
          cce::printf("[");
          int64_t base = i0 * stride0 + i1 * stride1 + i2 * stride2;
          for (int64_t i3 = 0; i3 < size3 - 1; i3++) {
            int64_t i = base + i3 * stride3;
            val = cast2f32(arg_ptr[i]);
            cce::printf(nlast_elem_fmt, val);
          }
          int64_t i = base + (size3 - 1) * stride3;
          val = cast2f32(arg_ptr[i]);
          cce::printf(last_elem_fmt, val);
          cce::printf(i2 < size2 - 1 ? "],\n" : "]");
        }
        cce::printf("]\n");
      }
    }
  } else if constexpr (std::is_same<T, bool>::value) {
    auto *byte_ptr = cast_bool_to_uint8_t(arg_ptr);
    for (int64_t i0 = 0; i0 < size0; i0++) {
      for (int64_t i1 = 0; i1 < size1; i1++) {
        cce::printf("(%d, %d, :, :):\n", i0, i1);
        cce::printf("[");
        for (int64_t i2 = 0; i2 < size2; i2++) {
          cce::printf("[");
          for (int64_t i3 = 0; i3 < size3; i3++) {
            int64_t i =
                i0 * stride0 + i1 * stride1 + i2 * stride2 + i3 * stride3;
            int64_t byte_offset = i / 8;
            int64_t bit_offset = i % 8;
            char val = ((byte_ptr[byte_offset]) >> bit_offset) & 1 ? 'T' : 'F';
            cce::printf(i3 < size3 - 1 ? nlast_elem_fmt : last_elem_fmt, val);
          }
          cce::printf(i2 < size2 - 1 ? "],\n" : "]");
        }
        cce::printf("]\n");
      }
    }
  } else {
    for (int64_t i0 = 0; i0 < size0; i0++) {
      for (int64_t i1 = 0; i1 < size1; i1++) {
        cce::printf("(%d, %d, :, :):\n", i0, i1);
        cce::printf("[");
        for (int64_t i2 = 0; i2 < size2; i2++) {
          cce::printf("[");
          int64_t base = i0 * stride0 + i1 * stride1 + i2 * stride2;
          for (int64_t i3 = 0; i3 < size3 - 1; i3++) {
            int64_t i = base + i3 * stride3;
            cce::printf(nlast_elem_fmt, arg_ptr[i]);
          }
          int64_t i = base + (size3 - 1) * stride3;
          cce::printf(last_elem_fmt, arg_ptr[i]);
          cce::printf(i2 < size2 - 1 ? "],\n" : "]");
        }
        cce::printf("]\n");
      }
    }
  }
}

template <typename T, typename MEM_T>
[aicore] __attribute__((always_inline)) void
print_5d_core(char *prefix, const int64_t len, memref_t<MEM_T, 5> *arg,
              const int8_t hex) {
  pipe_barrier(PIPE_ALL);

  const __gm__ char *nlast_elem_fmt;
  const __gm__ char *last_elem_fmt;
  print_core_prefix_fmt<T>(prefix, len, hex, &nlast_elem_fmt, &last_elem_fmt);

  auto arg_ptr = arg->aligned + arg->offset;
  const int64_t size0 = arg->sizes[0];
  const int64_t stride0 = arg->strides[0];
  const int64_t size1 = arg->sizes[1];
  const int64_t stride1 = arg->strides[1];
  const int64_t size2 = arg->sizes[2];
  const int64_t stride2 = arg->strides[2];
  const int64_t size3 = arg->sizes[3];
  const int64_t stride3 = arg->strides[3];
  const int64_t size4 = arg->sizes[4];
  const int64_t stride4 = arg->strides[4];
  if constexpr (std::is_same<T, half>::value ||
                std::is_same<T, bfloat16_t>::value ||
                std::is_same<T, float>::value) {
    float val;
    for (int64_t i0 = 0; i0 < size0; i0++) {
      for (int64_t i1 = 0; i1 < size1; i1++) {
        for (int64_t i2 = 0; i2 < size2; i2++) {
          cce::printf("(%d, %d, %d, :, :):\n", i0, i1, i2);
          cce::printf("[");
          for (int64_t i3 = 0; i3 < size3; i3++) {
            cce::printf("[");
            int64_t base =
                i0 * stride0 + i1 * stride1 + i2 * stride2 + i3 * stride3;
            for (int64_t i4 = 0; i4 < size4 - 1; i4++) {
              int64_t i = base + i4 * stride4;
              val = cast2f32(arg_ptr[i]);
              cce::printf(nlast_elem_fmt, val);
            }
            int64_t i = base + (size4 - 1) * stride4;
            val = cast2f32(arg_ptr[i]);
            cce::printf(last_elem_fmt, val);
            cce::printf(i3 < size3 - 1 ? "],\n" : "]");
          }
          cce::printf("]\n");
        }
      }
    }
  } else if constexpr (std::is_same<T, bool>::value) {
    auto *byte_ptr = cast_bool_to_uint8_t(arg_ptr);
    for (int64_t i0 = 0; i0 < size0; i0++) {
      for (int64_t i1 = 0; i1 < size1; i1++) {
        for (int64_t i2 = 0; i2 < size2; i2++) {
          cce::printf("(%d, %d, %d, :, :):\n", i0, i1, i2);
          cce::printf("[");
          for (int64_t i3 = 0; i3 < size3; i3++) {
            cce::printf("[");
            for (int64_t i4 = 0; i4 < size4; i4++) {
              int64_t i = i0 * stride0 + i1 * stride1 + i2 * stride2 +
                          i3 * stride3 + i4 * stride4;
              int64_t byte_offset = i / 8;
              int64_t bit_offset = i % 8;
              char val =
                  ((byte_ptr[byte_offset]) >> bit_offset) & 1 ? 'T' : 'F';
              cce::printf(i4 < size4 - 1 ? nlast_elem_fmt : last_elem_fmt, val);
            }
            cce::printf(i3 < size3 - 1 ? "],\n" : "]");
          }
          cce::printf("]\n");
        }
      }
    }
  } else {
    for (int64_t i0 = 0; i0 < size0; i0++) {
      for (int64_t i1 = 0; i1 < size1; i1++) {
        for (int64_t i2 = 0; i2 < size2; i2++) {
          cce::printf("(%d, %d, %d, :, :):\n", i0, i1, i2);
          cce::printf("[");
          for (int64_t i3 = 0; i3 < size3; i3++) {
            cce::printf("[");
            int64_t base =
                i0 * stride0 + i1 * stride1 + i2 * stride2 + i3 * stride3;
            for (int64_t i4 = 0; i4 < size4 - 1; i4++) {
              int64_t i = base + i4 * stride4;
              cce::printf(nlast_elem_fmt, arg_ptr[i]);
            }
            int64_t i = base + (size4 - 1) * stride4;
            cce::printf(last_elem_fmt, arg_ptr[i]);
            cce::printf(i3 < size3 - 1 ? "],\n" : "]");
          }
          cce::printf("]\n");
        }
      }
    }
  }
}

template <typename T, typename MEM_T>
[aicore] __attribute__((always_inline)) void
print_6d_core(char *prefix, const int64_t len, memref_t<MEM_T, 6> *arg,
              const int8_t hex) {
  pipe_barrier(PIPE_ALL);
  const __gm__ char *nlast_elem_fmt;
  const __gm__ char *last_elem_fmt;
  print_core_prefix_fmt<T>(prefix, len, hex, &nlast_elem_fmt, &last_elem_fmt);

  auto arg_ptr = arg->aligned + arg->offset;
  const int64_t size0 = arg->sizes[0];
  const int64_t stride0 = arg->strides[0];
  const int64_t size1 = arg->sizes[1];
  const int64_t stride1 = arg->strides[1];
  const int64_t size2 = arg->sizes[2];
  const int64_t stride2 = arg->strides[2];
  const int64_t size3 = arg->sizes[3];
  const int64_t stride3 = arg->strides[3];
  const int64_t size4 = arg->sizes[4];
  const int64_t stride4 = arg->strides[4];
  const int64_t size5 = arg->sizes[5];
  const int64_t stride5 = arg->strides[5];

  if constexpr (std::is_same<T, half>::value ||
                std::is_same<T, bfloat16_t>::value ||
                std::is_same<T, float>::value) {
    float val;
    for (int64_t i0 = 0; i0 < size0; i0++) {
      for (int64_t i1 = 0; i1 < size1; i1++) {
        for (int64_t i2 = 0; i2 < size2; i2++) {
          for (int64_t i3 = 0; i3 < size3; i3++) {
            cce::printf("(%d, %d, %d, %d, :, :):\n", i0, i1, i2, i3);
            cce::printf("[");
            for (int64_t i4 = 0; i4 < size4; i4++) {
              cce::printf("[");
              int64_t base = i0 * stride0 + i1 * stride1 + i2 * stride2 +
                             i3 * stride3 + i4 * stride4;
              for (int64_t i5 = 0; i5 < size5 - 1; i5++) {
                int64_t i = base + i5 * stride5;
                val = cast2f32(arg_ptr[i]);
                cce::printf(nlast_elem_fmt, val);
              }
              int64_t i = base + (size5 - 1) * stride5;
              val = cast2f32(arg_ptr[i]);
              cce::printf(last_elem_fmt, val);
              cce::printf(i4 < size4 - 1 ? "],\n" : "]");
            }
            cce::printf("]\n");
          }
        }
      }
    }
  } else if constexpr (std::is_same<T, bool>::value) {
    auto *byte_ptr = cast_bool_to_uint8_t(arg_ptr);
    for (int64_t i0 = 0; i0 < size0; i0++) {
      for (int64_t i1 = 0; i1 < size1; i1++) {
        for (int64_t i2 = 0; i2 < size2; i2++) {
          for (int64_t i3 = 0; i3 < size3; i3++) {
            cce::printf("(%d, %d, %d, %d, :, :):\n", i0, i1, i2, i3);
            cce::printf("[");
            for (int64_t i4 = 0; i4 < size4; i4++) {
              cce::printf("[");
              for (int64_t i5 = 0; i5 < size5; i5++) {
                int64_t i = i0 * stride0 + i1 * stride1 + i2 * stride2 +
                            i3 * stride3 + i4 * stride4 + i5 * stride5;
                int64_t byte_offset = i / 8;
                int64_t bit_offset = i % 8;
                char val =
                    ((byte_ptr[byte_offset]) >> bit_offset) & 1 ? 'T' : 'F';
                cce::printf(i5 < size5 - 1 ? nlast_elem_fmt : last_elem_fmt,
                            val);
              }
              cce::printf(i4 < size4 - 1 ? "],\n" : "]");
            }
            cce::printf("]\n");
          }
        }
      }
    }
  } else {
    for (int64_t i0 = 0; i0 < size0; i0++) {
      for (int64_t i1 = 0; i1 < size1; i1++) {
        for (int64_t i2 = 0; i2 < size2; i2++) {
          for (int64_t i3 = 0; i3 < size3; i3++) {
            cce::printf("(%d, %d, %d, %d, :, :):\n", i0, i1, i2, i3);
            cce::printf("[");
            for (int64_t i4 = 0; i4 < size4; i4++) {
              cce::printf("[");
              int64_t base = i0 * stride0 + i1 * stride1 + i2 * stride2 +
                             i3 * stride3 + i4 * stride4;
              for (int64_t i5 = 0; i5 < size5 - 1; i5++) {
                int64_t i = base + i5 * stride5;
                cce::printf(nlast_elem_fmt, arg_ptr[i]);
              }
              int64_t i = base + (size5 - 1) * stride5;
              cce::printf(last_elem_fmt, arg_ptr[i]);
              cce::printf(i4 < size4 - 1 ? "],\n" : "]");
            }
            cce::printf("]\n");
          }
        }
      }
    }
  }
}

template <typename T, typename MEM_T>
[aicore] __attribute__((always_inline)) void
print_7d_core(char *prefix, const int64_t len, memref_t<MEM_T, 7> *arg,
              const int8_t hex) {
  pipe_barrier(PIPE_ALL);

  const __gm__ char *nlast_elem_fmt;
  const __gm__ char *last_elem_fmt;
  print_core_prefix_fmt<T>(prefix, len, hex, &nlast_elem_fmt, &last_elem_fmt);

  auto arg_ptr = arg->aligned + arg->offset;
  const int64_t size0 = arg->sizes[0];
  const int64_t stride0 = arg->strides[0];
  const int64_t size1 = arg->sizes[1];
  const int64_t stride1 = arg->strides[1];
  const int64_t size2 = arg->sizes[2];
  const int64_t stride2 = arg->strides[2];
  const int64_t size3 = arg->sizes[3];
  const int64_t stride3 = arg->strides[3];
  const int64_t size4 = arg->sizes[4];
  const int64_t stride4 = arg->strides[4];
  const int64_t size5 = arg->sizes[5];
  const int64_t stride5 = arg->strides[5];
  const int64_t size6 = arg->sizes[6];
  const int64_t stride6 = arg->strides[6];

  if constexpr (std::is_same<T, half>::value ||
                std::is_same<T, bfloat16_t>::value ||
                std::is_same<T, float>::value) {
    float val;
    for (int64_t i0 = 0; i0 < size0; i0++) {
      for (int64_t i1 = 0; i1 < size1; i1++) {
        for (int64_t i2 = 0; i2 < size2; i2++) {
          for (int64_t i3 = 0; i3 < size3; i3++) {
            for (int64_t i4 = 0; i4 < size4; i4++) {
              cce::printf("(%d, %d, %d, %d, %d, :, :):\n", i0, i1, i2, i3, i4);
              cce::printf("[");
              for (int64_t i5 = 0; i5 < size5; i5++) {
                cce::printf("[");
                int64_t base = i0 * stride0 + i1 * stride1 + i2 * stride2 +
                               i3 * stride3 + i4 * stride4 + i5 * stride5;
                for (int64_t i6 = 0; i6 < size6 - 1; i6++) {
                  int64_t i = base + i6 * stride6;
                  val = cast2f32(arg_ptr[i]);
                  cce::printf(nlast_elem_fmt, val);
                }
                int64_t i = base + (size6 - 1) * stride6;
                val = cast2f32(arg_ptr[i]);
                cce::printf(last_elem_fmt, val);
                cce::printf(i5 < size5 - 1 ? "],\n" : "]");
              }
              cce::printf("]\n");
            }
          }
        }
      }
    }
  } else if constexpr (std::is_same<T, bool>::value) {
    auto *byte_ptr = cast_bool_to_uint8_t(arg_ptr);
    for (int64_t i0 = 0; i0 < size0; i0++) {
      for (int64_t i1 = 0; i1 < size1; i1++) {
        for (int64_t i2 = 0; i2 < size2; i2++) {
          for (int64_t i3 = 0; i3 < size3; i3++) {
            for (int64_t i4 = 0; i4 < size4; i4++) {
              cce::printf("(%d, %d, %d, %d, %d, :, :):\n", i0, i1, i2, i3, i4);
              cce::printf("[");
              for (int64_t i5 = 0; i5 < size5; i5++) {
                cce::printf("[");
                for (int64_t i6 = 0; i6 < size6; i6++) {
                  int64_t i = i0 * stride0 + i1 * stride1 + i2 * stride2 +
                              i3 * stride3 + i4 * stride4 + i5 * stride5 +
                              i6 * stride6;
                  int64_t byte_offset = i / 8;
                  int64_t bit_offset = i % 8;
                  char val =
                      ((byte_ptr[byte_offset]) >> bit_offset) & 1 ? 'T' : 'F';
                  cce::printf(i6 < size6 - 1 ? nlast_elem_fmt : last_elem_fmt,
                              val);
                }
                cce::printf(i5 < size5 - 1 ? "],\n" : "]");
              }
              cce::printf("]\n");
            }
          }
        }
      }
    }
  } else {
    for (int64_t i0 = 0; i0 < size0; i0++) {
      for (int64_t i1 = 0; i1 < size1; i1++) {
        for (int64_t i2 = 0; i2 < size2; i2++) {
          for (int64_t i3 = 0; i3 < size3; i3++) {
            for (int64_t i4 = 0; i4 < size4; i4++) {
              cce::printf("(%d, %d, %d, %d, %d, :, :):\n", i0, i1, i2, i3, i4);
              cce::printf("[");
              for (int64_t i5 = 0; i5 < size5; i5++) {
                cce::printf("[");
                int64_t base = i0 * stride0 + i1 * stride1 + i2 * stride2 +
                               i3 * stride3 + i4 * stride4 + i5 * stride5;
                for (int64_t i6 = 0; i6 < size6 - 1; i6++) {
                  int64_t i = base + i6 * stride6;
                  cce::printf(nlast_elem_fmt, arg_ptr[i]);
                }
                int64_t i = base + (size6 - 1) * stride6;
                cce::printf(last_elem_fmt, arg_ptr[i]);
                cce::printf(i5 < size5 - 1 ? "],\n" : "]");
              }
              cce::printf("]\n");
            }
          }
        }
      }
    }
  }
}

template <typename T, typename MEM_T>
[aicore] __attribute__((always_inline)) void
print_8d_core(char *prefix, const int64_t len, memref_t<MEM_T, 8> *arg,
              const int8_t hex) {
  pipe_barrier(PIPE_ALL);

  const __gm__ char *nlast_elem_fmt;
  const __gm__ char *last_elem_fmt;
  print_core_prefix_fmt<T>(prefix, len, hex, &nlast_elem_fmt, &last_elem_fmt);

  auto arg_ptr = arg->aligned + arg->offset;
  const int64_t size0 = arg->sizes[0];
  const int64_t stride0 = arg->strides[0];
  const int64_t size1 = arg->sizes[1];
  const int64_t stride1 = arg->strides[1];
  const int64_t size2 = arg->sizes[2];
  const int64_t stride2 = arg->strides[2];
  const int64_t size3 = arg->sizes[3];
  const int64_t stride3 = arg->strides[3];
  const int64_t size4 = arg->sizes[4];
  const int64_t stride4 = arg->strides[4];
  const int64_t size5 = arg->sizes[5];
  const int64_t stride5 = arg->strides[5];
  const int64_t size6 = arg->sizes[6];
  const int64_t stride6 = arg->strides[6];
  const int64_t size7 = arg->sizes[7];
  const int64_t stride7 = arg->strides[7];

  if constexpr (std::is_same<T, half>::value ||
                std::is_same<T, bfloat16_t>::value ||
                std::is_same<T, float>::value) {
    float val;
    for (int64_t i0 = 0; i0 < size0; i0++) {
      for (int64_t i1 = 0; i1 < size1; i1++) {
        for (int64_t i2 = 0; i2 < size2; i2++) {
          for (int64_t i3 = 0; i3 < size3; i3++) {
            for (int64_t i4 = 0; i4 < size4; i4++) {
              for (int64_t i5 = 0; i5 < size5; i5++) {
                cce::printf("(%d, %d, %d, %d, %d, %d, :, :):\n", i0, i1, i2, i3,
                            i4, i5);
                cce::printf("[");
                for (int64_t i6 = 0; i6 < size6; i6++) {
                  cce::printf("[");
                  int64_t base = i0 * stride0 + i1 * stride1 + i2 * stride2 +
                                 i3 * stride3 + i4 * stride4 + i5 * stride5 +
                                 i6 * stride6;
                  for (int64_t i7 = 0; i7 < size7 - 1; i7++) {
                    int64_t i = base + i7 * stride7;
                    val = cast2f32(arg_ptr[i]);
                    cce::printf(nlast_elem_fmt, val);
                  }
                  int64_t i = base + (size7 - 1) * stride7;
                  val = cast2f32(arg_ptr[i]);
                  cce::printf(last_elem_fmt, val);
                  cce::printf(i6 < size6 - 1 ? "],\n" : "]");
                }
                cce::printf("]\n");
              }
            }
          }
        }
      }
    }
  } else if constexpr (std::is_same<T, bool>::value) {
    auto *byte_ptr = cast_bool_to_uint8_t(arg_ptr);
    for (int64_t i0 = 0; i0 < size0; i0++) {
      for (int64_t i1 = 0; i1 < size1; i1++) {
        for (int64_t i2 = 0; i2 < size2; i2++) {
          for (int64_t i3 = 0; i3 < size3; i3++) {
            for (int64_t i4 = 0; i4 < size4; i4++) {
              for (int64_t i5 = 0; i5 < size5; i5++) {
                cce::printf("(%d, %d, %d, %d, %d, %d, :, :):\n", i0, i1, i2, i3,
                            i4, i5);
                cce::printf("[");
                for (int64_t i6 = 0; i6 < size6; i6++) {
                  cce::printf("[");
                  for (int64_t i7 = 0; i7 < size7; i7++) {
                    int64_t i = i0 * stride0 + i1 * stride1 + i2 * stride2 +
                                i3 * stride3 + i4 * stride4 + i5 * stride5 +
                                i6 * stride6 + i7 * stride7;
                    int64_t byte_offset = i / 8;
                    int64_t bit_offset = i % 8;
                    char val =
                        ((byte_ptr[byte_offset]) >> bit_offset) & 1 ? 'T' : 'F';
                    cce::printf(i7 < size7 - 1 ? nlast_elem_fmt : last_elem_fmt,
                                val);
                  }
                  cce::printf(i6 < size6 - 1 ? "],\n" : "]");
                }
                cce::printf("]\n");
              }
            }
          }
        }
      }
    }
  } else {
    for (int64_t i0 = 0; i0 < size0; i0++) {
      for (int64_t i1 = 0; i1 < size1; i1++) {
        for (int64_t i2 = 0; i2 < size2; i2++) {
          for (int64_t i3 = 0; i3 < size3; i3++) {
            for (int64_t i4 = 0; i4 < size4; i4++) {
              for (int64_t i5 = 0; i5 < size5; i5++) {
                cce::printf("(%d, %d, %d, %d, %d, %d, :, :):\n", i0, i1, i2, i3,
                            i4, i5);
                cce::printf("[");
                for (int64_t i6 = 0; i6 < size6; i6++) {
                  cce::printf("[");
                  int64_t base = i0 * stride0 + i1 * stride1 + i2 * stride2 +
                                 i3 * stride3 + i4 * stride4 + i5 * stride5 +
                                 i6 * stride6;
                  for (int64_t i7 = 0; i7 < size7 - 1; i7++) {
                    int64_t i = base + i7 * stride7;
                    cce::printf(nlast_elem_fmt, arg_ptr[i]);
                  }
                  int64_t i = base + (size7 - 1) * stride7;
                  cce::printf(last_elem_fmt, arg_ptr[i]);
                  cce::printf(i6 < size6 - 1 ? "],\n" : "]");
                }
                cce::printf("]\n");
              }
            }
          }
        }
      }
    }
  }
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

template <typename MEM_T>
[aicore] __attribute__((always_inline)) void
assert_1d_core(char *prefix, const int64_t len, memref_t<MEM_T, 1> *arg) {
  pipe_barrier(PIPE_ALL);
  auto arg_ptr = arg->aligned + arg->offset;
  const int64_t size0 = arg->sizes[0];
  const int64_t stride0 = arg->strides[0];
  for (int64_t i = 0; i < size0; i++) {
    if (!arg_ptr[i * stride0]) {
      npuir_cce_assert(false, prefix, len);
    }
  }
}

template <typename MEM_T>
[aicore] __attribute__((always_inline)) void
assert_2d_core(char *prefix, const int64_t len, memref_t<MEM_T, 2> *arg) {
  pipe_barrier(PIPE_ALL);
  auto arg_ptr = arg->aligned + arg->offset;
  const int64_t size0 = arg->sizes[0];
  const int64_t stride0 = arg->strides[0];
  const int64_t size1 = arg->sizes[1];
  const int64_t stride1 = arg->strides[1];
  for (int64_t i0 = 0; i0 < size0; i0++) {
    int64_t base0 = i0 * stride0;
    for (int64_t i1 = 0; i1 < size1; i1++) {
      if (!arg_ptr[base0 + i1 * stride1]) {
        npuir_cce_assert(false, prefix, len);
      }
    }
  }
}

template <typename MEM_T>
[aicore] __attribute__((always_inline)) void
assert_3d_core(char *prefix, const int64_t len, memref_t<MEM_T, 3> *arg) {
  pipe_barrier(PIPE_ALL);
  auto arg_ptr = arg->aligned + arg->offset;
  const int64_t size0 = arg->sizes[0];
  const int64_t stride0 = arg->strides[0];
  const int64_t size1 = arg->sizes[1];
  const int64_t stride1 = arg->strides[1];
  const int64_t size2 = arg->sizes[2];
  const int64_t stride2 = arg->strides[2];
  for (int64_t i0 = 0; i0 < size0; i0++) {
    int64_t base0 = i0 * stride0;
    for (int64_t i1 = 0; i1 < size1; i1++) {
      int64_t base1 = base0 + i1 * stride1;
      for (int64_t i2 = 0; i2 < size2; i2++) {
        if (!arg_ptr[base1 + i2 * stride2]) {
          npuir_cce_assert(false, prefix, len);
        }
      }
    }
  }
}

template <typename MEM_T>
[aicore] __attribute__((always_inline)) void
assert_4d_core(char *prefix, const int64_t len, memref_t<MEM_T, 4> *arg) {
  pipe_barrier(PIPE_ALL);
  auto arg_ptr = arg->aligned + arg->offset;
  const int64_t size0 = arg->sizes[0];
  const int64_t stride0 = arg->strides[0];
  const int64_t size1 = arg->sizes[1];
  const int64_t stride1 = arg->strides[1];
  const int64_t size2 = arg->sizes[2];
  const int64_t stride2 = arg->strides[2];
  const int64_t size3 = arg->sizes[3];
  const int64_t stride3 = arg->strides[3];
  for (int64_t i0 = 0; i0 < size0; i0++) {
    int64_t base0 = i0 * stride0;
    for (int64_t i1 = 0; i1 < size1; i1++) {
      int64_t base1 = base0 + i1 * stride1;
      for (int64_t i2 = 0; i2 < size2; i2++) {
        int64_t base2 = base1 + i2 * stride2;
        for (int64_t i3 = 0; i3 < size3; i3++) {
          if (!arg_ptr[base2 + i3 * stride3]) {
            npuir_cce_assert(false, prefix, len);
          }
        }
      }
    }
  }
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
