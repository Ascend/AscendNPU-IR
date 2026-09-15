/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef HIVM_MLIR_TEMPLATE_NCHW2NC1HWC0_UTILS_H
#define HIVM_MLIR_TEMPLATE_NCHW2NC1HWC0_UTILS_H

#include "Utils.h"

#define DECLARE_NCHW2NC1HWC0(type)                                            \
  __aicore__ __attribute__((always_inline)) void                              \
      _mlir_ciface_nchw2nc1hwc0_##type(                                       \
          memref_t<__gm__ type, 4> *src, memref_t<__cbuf__ type, 5> *dst,      \
          int64_t groups)

#define REGISTER_NCHW2NC1HWC0(type)                                           \
  DECLARE_NCHW2NC1HWC0(type) {                                                \
    copy_gm_to_cbuf_nchw2nc1hwc0_core<type>(src, dst, groups);                \
  }

#if defined(__DAV_C310__)
extern "C" {
DECLARE_NCHW2NC1HWC0(half);
DECLARE_NCHW2NC1HWC0(float);
DECLARE_NCHW2NC1HWC0(bfloat16_t);
}
#endif

#endif // HIVM_MLIR_TEMPLATE_NCHW2NC1HWC0_UTILS_H
