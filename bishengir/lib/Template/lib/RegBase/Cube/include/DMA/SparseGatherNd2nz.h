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

#ifndef HIVM_MLIR_TEMPLATE_SPARSE_GATHER_TO_L1BUFFER_H
#define HIVM_MLIR_TEMPLATE_SPARSE_GATHER_TO_L1BUFFER_H

#include "Utils.h"

// sparse_gather_load_to_L1buffer
//
// The library call name is
//   sparse_gather_load_to_L1buffer_<dtype>_<itype>
// produced by CustomOp's builtin mangling for
// `__builtin_sparse_gather_load_to_L1buffer`; see
// getSparseGatherLoadToL1BufferLibraryCallName() in
// LibraryFunctionOpInterfaceImpl.cpp. Keep the two in sync.
//
// Parameters are in the op's operand order: every input, then the destination
// buffer last. That is the order hivm.hir.custom carries them (inputs ahead of
// its output) and the order the generic HIVM-to-standard lowering forwards, so
// no per-op reordering is needed.
//
//   base              GM base of the gathered tensor. The caller folds any
//                     head/group offset into this pointer, so `base` is a plain
//                     rank-1 GM address whose token rows are `strideToken`
//                     elements apart and `dimSize` elements long.
//   index             GM base of this row's packed region metadata. Entry `s`
//                     holds `region_offset | (region_valid << shift)`, or a
//                     negative sentinel for a padding slot. Entries are sorted
//                     by region_offset ascending, which is what makes
//                     neighbouring slots mergeable.
//   blockR            slots in the destination tile (16, sometimes 8 or 32).
//   validRegionCount  only the first min(blockR, validRegionCount) entries of
//                     `index` are read.
//   regionSize        tokens per region (8).
//   dimSize           elements per token (192).
//   strideToken       element stride between consecutive tokens of `base`.
//   regionValidShift  bit position of the valid-token count (24).
//   regionIdMask      mask selecting the region offset ((1 << 24) - 1).
//   l2CacheCtrlMode   MTE2 cache-control field, passed through untouched.
//   dst               L1 destination, zN fractal (n1, d1, d0, n0) with
//                     n0 * sizeof(dtype) = 32B, holding blockR * regionSize
//                     token rows. Same layout the nd2nz template produces.
//
// One operand per call, not a fused K+V call sharing one scan. Fusing was
// measured and loses: the call is a completion barrier on the cube, so the QK
// dot would wait for the V gather too, and the added fixpipe stall exceeds the
// scalar saving. Scalar is not this kernel's binding constraint.
//
// zN rather than ND so the tile is already in the layout the cube wants and can
// reach the mmad through L1 -> L0 on MTE1, all on the cube. An ND operand would
// stage on the vector core, which neither the L1 -> UB transfer nor the vector
// calling convention ("arguments live in UB") can express for a cube-produced
// L1 buffer.
#define DECLARE_SPARSE_GATHER_LOAD_TO_L1BUFFER(dtype, itype)                   \
  __aicore__ __attribute__((always_inline)) void                               \
      _mlir_ciface_sparse_gather_load_to_L1buffer_##dtype##_##itype(           \
          memref_t<__gm__ dtype, 1> *base, memref_t<__gm__ itype, 1> *index,   \
          int32_t blockR, int32_t validRegionCount, int32_t regionSize,        \
          int32_t dimSize, int64_t strideToken, int32_t regionValidShift,      \
          int32_t regionIdMask, int32_t l2CacheCtrlMode,                       \
          memref_t<__cbuf__ dtype, 4> *dst)

#define REGISTER_SPARSE_GATHER_LOAD_TO_L1BUFFER(dtype, itype)                  \
  DECLARE_SPARSE_GATHER_LOAD_TO_L1BUFFER(dtype, itype) {                       \
    sparse_gather_load_to_L1buffer_core<dtype, itype>(                         \
        dst, base, index, blockR, validRegionCount, regionSize, dimSize,       \
        strideToken, regionValidShift, regionIdMask, l2CacheCtrlMode);         \
  }

#endif // HIVM_MLIR_TEMPLATE_SPARSE_GATHER_TO_L1BUFFER_H
