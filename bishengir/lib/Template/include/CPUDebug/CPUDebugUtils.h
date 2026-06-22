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

#ifndef BISHENGIR_LIB_TEMPLATE_INCLUDE_CPUDEBUG_UTILS_H
#define BISHENGIR_LIB_TEMPLATE_INCLUDE_CPUDEBUG_UTILS_H

#include "iostream"
#include <assert.h>
#include <cstring>

// prcess cce special grammar
#ifdef ENABLE_CPU_TRACE_INTRINSIC
// remove cce special grammer
#define __aiv__
#define __gm__
#define __ubuf__
#define __ca__
#define __cb__
#define __cc__
#define __cbuf__
#define __aicore__

// define cce built-in type
struct Half {
  unsigned short num;
};
struct Bfloat16 {
  unsigned short num;
};
typedef Bfloat16 bfloat16_t;
typedef Half half;

typedef enum {
  EVENT_ID0 = 0,
  EVENT_ID1,
  EVENT_ID2,
  EVENT_ID3,
  EVENT_ID4,
  EVENT_ID5,
  EVENT_ID6,
  EVENT_ID7,
} event_t;

typedef enum {
  PIPE_S = 0,
  PIPE_V,
  PIPE_M,
  PIPE_MTE1,
  PIPE_MTE2,
  PIPE_MTE3,
  PIPE_ALL,
} pipe_t;

typedef enum {
  VALUE_INDEX = 0,
  INDEX_VALUE = 1,
  ONLY_VALUE = 2,
  ONLY_INDEX = 3,
} Order_t;

typedef enum {
  VA0 = 0,
  VA1,
  VA2,
  VA3,
  VA4,
  VA5,
  VA6,
  VA7,
} ub_addr8_t;

typedef enum {
  NoQuant = 0,
  F322F16 = 1,
  REQ8 = 9,
  F322BF16 = 16,
} QuantMode_t;

typedef enum {
  inc = 0,
  dec = 1,
} addr_cal_mode_t;

typedef enum {
  PAD_NONE  = 0,
  PAD_MODE1 = 1,
  PAD_MODE2 = 2,
  PAD_MODE3 = 3,
  PAD_MODE4 = 4,
  PAD_MODE5 = 5,
  PAD_MODE6 = 6,
  PAD_MODE7 = 7,
  PAD_MODE8 = 8,
} pad_t;
#endif

// define assert behavior
#ifdef ENABLE_CPU_TRACE_INTRINSIC
// Determine whether the size is 32byte aligned.
template <typename T> bool isSizeAlignedToBlock(int size) {
  auto num = 32 / sizeof(T);
  return size % num == 0;
}
#endif

// define intrinsic behavior
#ifdef ENABLE_CPU_TRACE_INTRINSIC
#define TO_STRING(NAME) #NAME
// Print cce hardware instructions in debug mode.
template <typename... ARGS>
void printIntrinsic(std::string intrinsicName, ARGS... args) {
  bool isFirstArgs = true;
  std::cout << intrinsicName;
  std::cout << "(";
  (void)std::initializer_list<std::string>{(
      [&args, &isFirstArgs] {
        if (isFirstArgs) {
          if constexpr (std::is_same<ARGS, bfloat16_t>::value ||
                        std::is_same<ARGS, half>::value) {
            std::cout << args.num;
          } else {
            std::cout << args;
          }
          isFirstArgs = false;
        } else {
          std::cout << ", ";
          if constexpr (std::is_same<ARGS, bfloat16_t>::value ||
                        std::is_same<ARGS, half>::value) {
            std::cout << args.num;
          } else {
            std::cout << args;
          }
        }
      }(),
      intrinsicName)...};
  std::cout << ");" << std::endl;
}
// Print hardware instructions with parameters.
#define INTRINSIC(NAME, ...) printIntrinsic(TO_STRING(NAME), __VA_ARGS__)
// Print hardware instructions without parameters.
#define INTRINSIC_NO_ARGS(NAME)                                                \
  std::cout << TO_STRING(NAME) << "();" << std::endl
#define GET_MAX_MIN_CNT() 0
#define GET_ACC_VAL() 0
#else
// Print hardware instructions with parameters.
#define INTRINSIC(NAME, ...) NAME(__VA_ARGS__)
// Print hardware instructions without parameters.
#define INTRINSIC_NO_ARGS(NAME) NAME()
#define GET_MAX_MIN_CNT() get_max_min_cnt()
#define GET_ACC_VAL() get_acc_val()
#endif

#endif // BISHENGIR_LIB_TEMPLATE_INCLUDE_CPUDEBUG_UTILS_H