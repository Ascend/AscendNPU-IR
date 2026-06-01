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

// SIMT indirect bitwise atomic with UB software acceleration.
//
// Background:
//   GM atomicAnd/atomicOr/atomicXOr do not have native hardware instructions
//   on this target and are implemented in software, so performing those atomic
//   operations directly in GM is slow. UB atomics are efficient inside a CTA.
//   This template handles default-scope tl.atomic_{or,and,xor} by moving the
//   hot per-CTA aggregation into UB. The aggregated CTA contribution is then
//   committed to GM with one CAS loop per touched in-range slot, which prevents
//   cross-CTA plain-store overwrite when grid size is larger than one.
//
// Temp buffer contract:
//   The lowering pass allocates one UB temp buffer, here called buffer. The
//   fast path can index buffer[offset] only when:
//
//       0 <= offset < bufferSize
//
//   Sparse GM offsets outside that range are legal indirect-atomic addresses,
//   but they are not legal temp-buffer indices. Those lanes take a local
//   fallback to the original GM atomic path instead of touching UB.
//
// Data layout example:
//
//     indicesSize = 4
//
//     lane id:        0    1    2    3
//     offsets:      [ 0,   3,   6,   3 ]
//     values:       [v0,  v1,  v2,  v3]
//                    |    |         |
//                    |    |         |
//                    v    v         v
//     UB temp:      [c0,  c1,  c2,  c3]
//                    ^              ^
//                    |              |
//                    |              +-- lanes 1/3 aggregate contribution c3
//                    +-- lane 0 aggregates contribution c0
//
//     GM src:       [g0,  g1,  g2,  g3,  g4,  g5,  g6,  ...]
//                    ^              ^              ^
//                    |              |              |
//       Soft CAS:    +-- slot 0     +-- slot 3     |
//       GM fallback:                                +-- lane 2, offset 6
//
//   In-range offsets, such as 0/3, are valid UB temp indices. Their
//   updates are first accumulated as per-CTA contributions by UB atomics and
//   then committed to the corresponding GM locations with CAS loops. Duplicated
//   in-range offsets, such as offset 3 above, only contend inside UB and issue
//   one GM CAS commit for that temp-buffer slot.
//
//   Out-of-range offsets, such as 6, are valid GM addresses but invalid UB
//   temp indices. Those lanes immediately perform GM atomic_{or,and,xor}, write
//   their returned old value to out, and skip the later UB aggregate/commit
//   phases.
//
// This keeps the common dense-offset case fast, avoids UB access overflow for
// sparse offsets, prevents a single sparse lane from forcing the whole CTA back
// to the slower GM atomic implementation, and avoids lost updates across CTAs.

#if defined(__DAV_C310__)

#include "__clang_cce_simt_intrinsics.h"
#include "RegBase/VecUtils.h"

constexpr unsigned int MAX_THREAD_NUM = 1024;

enum class IndirectAtomicSoftAccelOp { Or, And, Xor };

template <typename T>
inline constexpr bool kIsValidItype =
    std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>;

template <typename T>
inline constexpr bool kIsValidBitwiseAtomicDtype =
    std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> ||
    std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>;

template <typename ITYPE>
__simt_callee__ __aiv__ __attribute__((always_inline)) static bool
isOffsetInBuffer(ITYPE offset, const int64_t bufferSize) {
  return offset >= 0 && static_cast<uint64_t>(offset) <
                            static_cast<uint64_t>(bufferSize);
}

template <IndirectAtomicSoftAccelOp Op, typename DTYPE>
__simt_callee__ __aiv__ __attribute__((always_inline)) static DTYPE
atomicBitwiseUB(__ubuf__ DTYPE *addr, DTYPE val) {
  // UB atomics are available at 32-bit granularity. Split 64-bit bitwise
  // atomics into independent low/high-word updates; bitwise operations do not
  // carry between words, so this preserves the final value.
  if constexpr (std::is_same_v<DTYPE, uint64_t> ||
                std::is_same_v<DTYPE, int64_t>) {
    __ubuf__ uint32_t *addr32 = reinterpret_cast<__ubuf__ uint32_t *>(addr);
    uint64_t rawVal = static_cast<uint64_t>(val);
    uint32_t loVal = static_cast<uint32_t>(rawVal);
    uint32_t hiVal = static_cast<uint32_t>(rawVal >> 32);
    uint32_t loOld;
    uint32_t hiOld;
    if constexpr (Op == IndirectAtomicSoftAccelOp::Or) {
      loOld = atomicOr(addr32, loVal);
      hiOld = atomicOr(addr32 + 1, hiVal);
    } else if constexpr (Op == IndirectAtomicSoftAccelOp::And) {
      loOld = atomicAnd(addr32, loVal);
      hiOld = atomicAnd(addr32 + 1, hiVal);
    } else {
      loOld = atomicXOr(addr32, loVal);
      hiOld = atomicXOr(addr32 + 1, hiVal);
    }
    uint64_t rawOld = (static_cast<uint64_t>(hiOld) << 32) | loOld;
    return static_cast<DTYPE>(rawOld);
  } else if constexpr (Op == IndirectAtomicSoftAccelOp::Or) {
    return atomicOr(addr, val);
  } else if constexpr (Op == IndirectAtomicSoftAccelOp::And) {
    return atomicAnd(addr, val);
  } else {
    return atomicXOr(addr, val);
  }
}

template <IndirectAtomicSoftAccelOp Op, typename DTYPE>
__simt_callee__ __aiv__ __attribute__((always_inline)) static DTYPE
bitwiseIdentity() {
  if constexpr (Op == IndirectAtomicSoftAccelOp::And) {
    return static_cast<DTYPE>(-1);
  } else {
    return static_cast<DTYPE>(0);
  }
}

template <IndirectAtomicSoftAccelOp Op, typename DTYPE>
__simt_callee__ __aiv__ __attribute__((always_inline)) static DTYPE
applyBitwise(DTYPE lhs, DTYPE rhs) {
  if constexpr (Op == IndirectAtomicSoftAccelOp::Or) {
    return lhs | rhs;
  } else if constexpr (Op == IndirectAtomicSoftAccelOp::And) {
    return lhs & rhs;
  } else {
    return lhs ^ rhs;
  }
}

template <IndirectAtomicSoftAccelOp Op, typename DTYPE>
__simt_callee__ __aiv__ __attribute__((always_inline)) static DTYPE
atomicBitwiseGM(__gm__ DTYPE *addr, DTYPE val) {
  if constexpr (Op == IndirectAtomicSoftAccelOp::Or) {
    return atomicOr(addr, val);
  } else if constexpr (Op == IndirectAtomicSoftAccelOp::And) {
    return atomicAnd(addr, val);
  } else {
    return atomicXOr(addr, val);
  }
}

template <IndirectAtomicSoftAccelOp Op, typename DTYPE>
__simt_callee__ __aiv__ __attribute__((always_inline)) static DTYPE
commitBitwiseGMWithCAS(__gm__ DTYPE *addr, DTYPE contribution) {
  DTYPE oldValue = *addr;
  DTYPE assumed;
  do {
    assumed = oldValue;
    DTYPE newValue = applyBitwise<Op, DTYPE>(assumed, contribution);
    if (newValue == assumed)
      return oldValue;
    oldValue = atomicCAS(addr, assumed, newValue);
  } while (oldValue != assumed);
  return oldValue;
}

// Initialize UB accumulators with the operation identity before lanes merge
// their in-range updates through UB atomics. Sparse offsets outside the
// temp-buffer range locally fall back to the original GM atomic path.
template <IndirectAtomicSoftAccelOp Op, typename DTYPE, typename ITYPE>
__simt_callee__ __aiv__
    __attribute__((always_inline)) static void
    SimtIndirectAtomicSoftAccelLoad(__gm__ DTYPE *src, __ubuf__ ITYPE *offsets,
                              __ubuf__ DTYPE *value, __ubuf__ int8_t *mask,
                              __ubuf__ DTYPE *out, __ubuf__ DTYPE *buffer,
                              const int64_t indicesSize,
                              const int64_t bufferSize,
                              const int64_t offsetStride,
                              const int64_t valueStride,
                              const int64_t maskStride,
                              const int64_t outStride,
                              const int64_t bufferStride) {
  for (uint64_t idx = threadIdx.x; idx < static_cast<uint64_t>(bufferSize);
       idx += blockDim.x) {
    buffer[idx * bufferStride] = bitwiseIdentity<Op, DTYPE>();
  }

  for (uint64_t idx = threadIdx.x; idx < static_cast<uint64_t>(indicesSize);
       idx += blockDim.x) {
    if (!mask[idx * maskStride]) {
      out[idx * outStride] = MakeSentinelNegOne<DTYPE>();
      continue;
    }

    ITYPE offset = offsets[idx * offsetStride];
    if (!isOffsetInBuffer(offset, bufferSize)) {
      out[idx * outStride] =
          atomicBitwiseGM<Op, DTYPE>(src + offset, value[idx * valueStride]);
      continue;
    }

    DTYPE oldValue = src[offset];
    out[idx * outStride] = oldValue;
  }
}

template <IndirectAtomicSoftAccelOp Op, typename DTYPE, typename ITYPE>
__simt_callee__ __aiv__
    __attribute__((always_inline)) static void
    SimtIndirectAtomicSoftAccelLoadWithoutMask(__gm__ DTYPE *src,
                                         __ubuf__ ITYPE *offsets,
                                         __ubuf__ DTYPE *value,
                                         __ubuf__ DTYPE *out,
                                         __ubuf__ DTYPE *buffer,
                                         const int64_t indicesSize,
                                         const int64_t bufferSize,
                                         const int64_t offsetStride,
                                         const int64_t valueStride,
                                         const int64_t outStride,
                                         const int64_t bufferStride) {
  for (uint64_t idx = threadIdx.x; idx < static_cast<uint64_t>(bufferSize);
       idx += blockDim.x) {
    buffer[idx * bufferStride] = bitwiseIdentity<Op, DTYPE>();
  }

  for (uint64_t idx = threadIdx.x; idx < static_cast<uint64_t>(indicesSize);
       idx += blockDim.x) {
    ITYPE offset = offsets[idx * offsetStride];
    if (!isOffsetInBuffer(offset, bufferSize)) {
      out[idx * outStride] =
          atomicBitwiseGM<Op, DTYPE>(src + offset, value[idx * valueStride]);
      continue;
    }

    DTYPE oldValue = src[offset];
    out[idx * outStride] = oldValue;
  }
}

// Duplicate in-range offsets race only inside UB here. Out-of-range offsets
// were already handled by the local GM fallback in the load phase.
template <IndirectAtomicSoftAccelOp Op, typename DTYPE, typename ITYPE>
__simt_callee__ __aiv__
    __attribute__((always_inline)) static void
    SimtIndirectAtomicSoftAccelAggregate(__ubuf__ ITYPE *offsets,
                                   __ubuf__ DTYPE *value,
                                   __ubuf__ int8_t *mask,
                                   __ubuf__ DTYPE *buffer,
                                   const int64_t indicesSize,
                                   const int64_t bufferSize,
                                   const int64_t offsetStride,
                                   const int64_t valueStride,
                                   const int64_t maskStride,
                                   const int64_t bufferStride) {
  for (uint64_t idx = threadIdx.x; idx < static_cast<uint64_t>(indicesSize);
       idx += blockDim.x) {
    if (!mask[idx * maskStride])
      continue;

    ITYPE offset = offsets[idx * offsetStride];
    if (!isOffsetInBuffer(offset, bufferSize))
      continue;
    atomicBitwiseUB<Op, DTYPE>(buffer + offset * bufferStride,
                               value[idx * valueStride]);
  }
}

template <IndirectAtomicSoftAccelOp Op, typename DTYPE, typename ITYPE>
__simt_callee__ __aiv__
    __attribute__((always_inline)) static void
    SimtIndirectAtomicSoftAccelAggregateWithoutMask(__ubuf__ ITYPE *offsets,
                                              __ubuf__ DTYPE *value,
                                              __ubuf__ DTYPE *buffer,
                                              const int64_t indicesSize,
                                              const int64_t bufferSize,
                                              const int64_t offsetStride,
                                              const int64_t valueStride,
                                              const int64_t bufferStride) {
  for (uint64_t idx = threadIdx.x; idx < static_cast<uint64_t>(indicesSize);
       idx += blockDim.x) {
    ITYPE offset = offsets[idx * offsetStride];
    if (!isOffsetInBuffer(offset, bufferSize))
      continue;
    atomicBitwiseUB<Op, DTYPE>(buffer + offset * bufferStride,
                               value[idx * valueStride]);
  }
}

template <IndirectAtomicSoftAccelOp Op, typename DTYPE>
__simt_callee__ __aiv__
    __attribute__((always_inline)) static void
    SimtIndirectAtomicSoftAccelCommit(__gm__ DTYPE *src, __ubuf__ DTYPE *buffer,
                                const int64_t bufferSize,
                                const int64_t bufferStride) {
  const DTYPE identity = bitwiseIdentity<Op, DTYPE>();
  for (uint64_t slot = threadIdx.x; slot < static_cast<uint64_t>(bufferSize);
       slot += blockDim.x) {
    DTYPE contribution = buffer[slot * bufferStride];
    if (contribution == identity)
      continue;
    commitBitwiseGMWithCAS<Op, DTYPE>(src + slot, contribution);
  }
}

template <IndirectAtomicSoftAccelOp Op, typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM) __aiv__
    __attribute__((always_inline)) static void
    SimtIndirectAtomicSoftAccelLoadKernel(__gm__ DTYPE *src,
                                    __ubuf__ ITYPE *offsets,
                                    __ubuf__ DTYPE *value,
                                    __ubuf__ int8_t *mask,
                                    __ubuf__ DTYPE *out,
                                    __ubuf__ DTYPE *buffer,
                                    const int64_t indicesSize,
                                    const int64_t bufferSize,
                                    const int64_t offsetStride,
                                    const int64_t valueStride,
                                    const int64_t maskStride,
                                    const int64_t outStride,
                                    const int64_t bufferStride) {
  SimtIndirectAtomicSoftAccelLoad<Op, DTYPE, ITYPE>(
      src, offsets, value, mask, out, buffer, indicesSize, bufferSize, offsetStride,
      valueStride, maskStride, outStride, bufferStride);
}

template <IndirectAtomicSoftAccelOp Op, typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM) __aiv__
    __attribute__((always_inline)) static void
    SimtIndirectAtomicSoftAccelAggregateKernel(
        __ubuf__ ITYPE *offsets, __ubuf__ DTYPE *value, __ubuf__ int8_t *mask,
        __ubuf__ DTYPE *buffer, const int64_t indicesSize,
        const int64_t bufferSize, const int64_t offsetStride, const int64_t valueStride,
        const int64_t maskStride, const int64_t bufferStride) {
  SimtIndirectAtomicSoftAccelAggregate<Op, DTYPE, ITYPE>(
      offsets, value, mask, buffer, indicesSize, bufferSize, offsetStride, valueStride,
      maskStride, bufferStride);
}

template <IndirectAtomicSoftAccelOp Op, typename DTYPE>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM) __aiv__
    __attribute__((always_inline)) static void
    SimtIndirectAtomicSoftAccelCommitKernel(__gm__ DTYPE *src,
                                      __ubuf__ DTYPE *buffer,
                                      const int64_t bufferSize,
                                      const int64_t bufferStride) {
  SimtIndirectAtomicSoftAccelCommit<Op, DTYPE>(src, buffer, bufferSize,
                                         bufferStride);
}

template <IndirectAtomicSoftAccelOp Op, typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM) __aiv__
    __attribute__((always_inline)) static void
    SimtIndirectAtomicSoftAccelLoadKernelWithoutMask(
        __gm__ DTYPE *src, __ubuf__ ITYPE *offsets, __ubuf__ DTYPE *value,
        __ubuf__ DTYPE *out, __ubuf__ DTYPE *buffer,
        const int64_t indicesSize, const int64_t bufferSize,
        const int64_t offsetStride,
        const int64_t valueStride, const int64_t outStride,
        const int64_t bufferStride) {
  SimtIndirectAtomicSoftAccelLoadWithoutMask<Op, DTYPE, ITYPE>(
      src, offsets, value, out, buffer, indicesSize, bufferSize, offsetStride,
      valueStride, outStride, bufferStride);
}

template <IndirectAtomicSoftAccelOp Op, typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM) __aiv__
    __attribute__((always_inline)) static void
    SimtIndirectAtomicSoftAccelAggregateKernelWithoutMask(
        __ubuf__ ITYPE *offsets, __ubuf__ DTYPE *value,
        __ubuf__ DTYPE *buffer, const int64_t indicesSize,
        const int64_t bufferSize, const int64_t offsetStride, const int64_t valueStride,
        const int64_t bufferStride) {
  SimtIndirectAtomicSoftAccelAggregateWithoutMask<Op, DTYPE, ITYPE>(
      offsets, value, buffer, indicesSize, bufferSize, offsetStride, valueStride,
      bufferStride);
}

template <IndirectAtomicSoftAccelOp Op, typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM) __aiv__
    __attribute__((always_inline)) static void
    SimtIndirectAtomicSoftAccelKernel(__gm__ DTYPE *src, __ubuf__ ITYPE *offsets,
                                __ubuf__ DTYPE *value, __ubuf__ int8_t *mask,
                                __ubuf__ DTYPE *out, __ubuf__ DTYPE *buffer,
                                const int64_t indicesSize,
                                const int64_t bufferSize,
                                const int64_t offsetStride,
                                const int64_t valueStride,
                                const int64_t maskStride,
                                const int64_t outStride,
                                const int64_t bufferStride) {
  SimtIndirectAtomicSoftAccelLoad<Op, DTYPE, ITYPE>(
      src, offsets, value, mask, out, buffer, indicesSize, bufferSize, offsetStride,
      valueStride, maskStride, outStride, bufferStride);
  SimtIndirectAtomicSoftAccelAggregate<Op, DTYPE, ITYPE>(
      offsets, value, mask, buffer, indicesSize, bufferSize, offsetStride, valueStride,
      maskStride, bufferStride);
  SimtIndirectAtomicSoftAccelCommit<Op, DTYPE>(src, buffer, bufferSize,
                                         bufferStride);
}

template <IndirectAtomicSoftAccelOp Op, typename DTYPE, typename ITYPE>
__simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM) __aiv__
    __attribute__((always_inline)) static void
    SimtIndirectAtomicSoftAccelKernelWithoutMask(
        __gm__ DTYPE *src, __ubuf__ ITYPE *offsets, __ubuf__ DTYPE *value,
        __ubuf__ DTYPE *out, __ubuf__ DTYPE *buffer,
        const int64_t indicesSize, const int64_t bufferSize,
        const int64_t offsetStride,
        const int64_t valueStride, const int64_t outStride,
        const int64_t bufferStride) {
  SimtIndirectAtomicSoftAccelLoadWithoutMask<Op, DTYPE, ITYPE>(
      src, offsets, value, out, buffer, indicesSize, bufferSize, offsetStride,
      valueStride, outStride, bufferStride);
  SimtIndirectAtomicSoftAccelAggregateWithoutMask<Op, DTYPE, ITYPE>(
      offsets, value, buffer, indicesSize, bufferSize, offsetStride, valueStride,
      bufferStride);
  SimtIndirectAtomicSoftAccelCommit<Op, DTYPE>(src, buffer, bufferSize,
                                         bufferStride);
}

template <IndirectAtomicSoftAccelOp Op, typename DTYPE, typename ITYPE>
__aiv__ __attribute__((always_inline)) static void indirect_atomic_soft_impl(
    memref_t<__gm__ DTYPE, 1> *src, memref_t<__ubuf__ ITYPE, 1> *offsets,
    memref_t<__ubuf__ DTYPE, 1> *value, memref_t<__ubuf__ int8_t, 1> *mask,
    memref_t<__ubuf__ DTYPE, 1> *out, memref_t<__ubuf__ DTYPE, 1> *buffer) {
  auto srcPtr = reinterpret_cast<__gm__ DTYPE *>(src->aligned);
  auto offsetsPtr =
      reinterpret_cast<__ubuf__ ITYPE *>(offsets->aligned + offsets->offset);
  auto valuePtr =
      reinterpret_cast<__ubuf__ DTYPE *>(value->aligned + value->offset);
  auto maskPtr =
      reinterpret_cast<__ubuf__ int8_t *>(mask->aligned + mask->offset);
  auto outPtr = reinterpret_cast<__ubuf__ DTYPE *>(out->aligned + out->offset);
  auto bufferPtr =
      reinterpret_cast<__ubuf__ DTYPE *>(buffer->aligned + buffer->offset);

  // Launch phases separately to give the SIMT runtime a hard ordering point
  // between UB initialization, UB atomic aggregation, and GM CAS commit.
  cce::async_invoke<SimtIndirectAtomicSoftAccelLoadKernel<Op, DTYPE, ITYPE>>(
      cce::dim3{MAX_THREAD_NUM}, srcPtr, offsetsPtr, valuePtr, maskPtr, outPtr,
      bufferPtr,
      offsets->sizes[0], buffer->sizes[0], offsets->strides[0], value->strides[0],
      mask->strides[0], out->strides[0], buffer->strides[0]);
  cce::async_invoke<SimtIndirectAtomicSoftAccelAggregateKernel<Op, DTYPE, ITYPE>>(
      cce::dim3{MAX_THREAD_NUM}, offsetsPtr, valuePtr, maskPtr, bufferPtr,
      offsets->sizes[0], buffer->sizes[0], offsets->strides[0], value->strides[0],
      mask->strides[0], buffer->strides[0]);
  cce::async_invoke<SimtIndirectAtomicSoftAccelCommitKernel<Op, DTYPE>>(
      cce::dim3{MAX_THREAD_NUM}, srcPtr, bufferPtr, buffer->sizes[0],
      buffer->strides[0]);
}

template <IndirectAtomicSoftAccelOp Op, typename DTYPE, typename ITYPE>
__aiv__ __attribute__((always_inline)) static void
indirect_atomic_soft_no_mask_impl(
    memref_t<__gm__ DTYPE, 1> *src, memref_t<__ubuf__ ITYPE, 1> *offsets,
    memref_t<__ubuf__ DTYPE, 1> *value, memref_t<__ubuf__ DTYPE, 1> *out,
    memref_t<__ubuf__ DTYPE, 1> *buffer) {
  auto srcPtr = reinterpret_cast<__gm__ DTYPE *>(src->aligned);
  auto offsetsPtr =
      reinterpret_cast<__ubuf__ ITYPE *>(offsets->aligned + offsets->offset);
  auto valuePtr =
      reinterpret_cast<__ubuf__ DTYPE *>(value->aligned + value->offset);
  auto outPtr = reinterpret_cast<__ubuf__ DTYPE *>(out->aligned + out->offset);
  auto bufferPtr =
      reinterpret_cast<__ubuf__ DTYPE *>(buffer->aligned + buffer->offset);

  cce::async_invoke<SimtIndirectAtomicSoftAccelLoadKernelWithoutMask<Op, DTYPE,
                                                               ITYPE>>(
      cce::dim3{MAX_THREAD_NUM}, srcPtr, offsetsPtr, valuePtr, outPtr,
      bufferPtr, offsets->sizes[0], buffer->sizes[0], offsets->strides[0],
      value->strides[0], out->strides[0], buffer->strides[0]);
  cce::async_invoke<SimtIndirectAtomicSoftAccelAggregateKernelWithoutMask<
      Op, DTYPE, ITYPE>>(cce::dim3{MAX_THREAD_NUM}, offsetsPtr, valuePtr,
                         bufferPtr, offsets->sizes[0], buffer->sizes[0],
                         offsets->strides[0], value->strides[0],
                         buffer->strides[0]);
  cce::async_invoke<SimtIndirectAtomicSoftAccelCommitKernel<Op, DTYPE>>(
      cce::dim3{MAX_THREAD_NUM}, srcPtr, bufferPtr, buffer->sizes[0],
      buffer->strides[0]);
}

#define DEFINE_INDIRECT_ATOMIC_SOFT_WRAPPER(op, opEnum)                        \
  template <typename DTYPE, typename ITYPE,                                   \
            std::enable_if_t<kIsValidBitwiseAtomicDtype<DTYPE> &&             \
                                 kIsValidItype<ITYPE>,                       \
                             int> = 0>                                        \
  __aiv__ __attribute__((always_inline)) void indirect_atomic_soft_##op(       \
      memref_t<__gm__ DTYPE, 1> *src,                                         \
      memref_t<__ubuf__ ITYPE, 1> *offsets,                                   \
      memref_t<__ubuf__ DTYPE, 1> *value,                                     \
      memref_t<__ubuf__ int8_t, 1> *mask,                                     \
      memref_t<__ubuf__ DTYPE, 1> *out,                                       \
      memref_t<__ubuf__ DTYPE, 1> *buffer) {                                  \
    indirect_atomic_soft_impl<IndirectAtomicSoftAccelOp::opEnum, DTYPE, ITYPE>(      \
        src, offsets, value, mask, out, buffer);                              \
  }                                                                           \
  template <typename DTYPE, typename ITYPE,                                   \
            std::enable_if_t<kIsValidBitwiseAtomicDtype<DTYPE> &&             \
                                 kIsValidItype<ITYPE>,                       \
                             int> = 0>                                        \
  __aiv__ __attribute__((always_inline)) void                                 \
  indirect_atomic_soft_##op##_no_mask(                                         \
      memref_t<__gm__ DTYPE, 1> *src,                                         \
      memref_t<__ubuf__ ITYPE, 1> *offsets,                                   \
      memref_t<__ubuf__ DTYPE, 1> *value,                                     \
      memref_t<__ubuf__ DTYPE, 1> *out,                                       \
      memref_t<__ubuf__ DTYPE, 1> *buffer) {                                  \
    indirect_atomic_soft_no_mask_impl<IndirectAtomicSoftAccelOp::opEnum, DTYPE,      \
                                     ITYPE>(src, offsets, value, out,         \
                                            buffer);                          \
  }

DEFINE_INDIRECT_ATOMIC_SOFT_WRAPPER(or, Or)
DEFINE_INDIRECT_ATOMIC_SOFT_WRAPPER(and, And)
DEFINE_INDIRECT_ATOMIC_SOFT_WRAPPER(xor, Xor)

#undef DEFINE_INDIRECT_ATOMIC_SOFT_WRAPPER

#define REGISTE_INDIRECT_ATOMIC_SOFT(op, dtype, itype)                         \
  __aiv__ __attribute__((always_inline)) void                                 \
      _mlir_ciface_indirect_atomic_soft_##op##_##dtype##_##itype(             \
          memref_t<__gm__ dtype, 1> *src,                                     \
          memref_t<__ubuf__ itype, 1> *offsets,                               \
          memref_t<__ubuf__ dtype, 1> *value,                                 \
          memref_t<__ubuf__ int8_t, 1> *mask,                                 \
          memref_t<__ubuf__ dtype, 1> *out,                                   \
          memref_t<__ubuf__ dtype, 1> *buffer) {                              \
    indirect_atomic_soft_##op<dtype, itype>(src, offsets, value, mask, out,    \
                                           buffer);                           \
  }

#define REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(op, dtype, itype)                 \
  __aiv__ __attribute__((always_inline)) void                                 \
      _mlir_ciface_indirect_atomic_soft_##op##_no_mask_##dtype##_##itype(      \
          memref_t<__gm__ dtype, 1> *src,                                     \
          memref_t<__ubuf__ itype, 1> *offsets,                               \
          memref_t<__ubuf__ dtype, 1> *value,                                 \
          memref_t<__ubuf__ dtype, 1> *out,                                   \
          memref_t<__ubuf__ dtype, 1> *buffer) {                              \
    indirect_atomic_soft_##op##_no_mask<dtype, itype>(src, offsets, value,     \
                                                     out, buffer);            \
  }

extern "C" {
REGISTE_INDIRECT_ATOMIC_SOFT(or, int32_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT(or, int64_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT(or, uint32_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT(or, uint64_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT(or, int32_t, int64_t);
REGISTE_INDIRECT_ATOMIC_SOFT(or, int64_t, int64_t);
REGISTE_INDIRECT_ATOMIC_SOFT(or, uint32_t, int64_t);
REGISTE_INDIRECT_ATOMIC_SOFT(or, uint64_t, int64_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(or, int32_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(or, int64_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(or, uint32_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(or, uint64_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(or, int32_t, int64_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(or, int64_t, int64_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(or, uint32_t, int64_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(or, uint64_t, int64_t);

REGISTE_INDIRECT_ATOMIC_SOFT(and, int32_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT(and, int64_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT(and, uint32_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT(and, uint64_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT(and, int32_t, int64_t);
REGISTE_INDIRECT_ATOMIC_SOFT(and, int64_t, int64_t);
REGISTE_INDIRECT_ATOMIC_SOFT(and, uint32_t, int64_t);
REGISTE_INDIRECT_ATOMIC_SOFT(and, uint64_t, int64_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(and, int32_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(and, int64_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(and, uint32_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(and, uint64_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(and, int32_t, int64_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(and, int64_t, int64_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(and, uint32_t, int64_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(and, uint64_t, int64_t);

REGISTE_INDIRECT_ATOMIC_SOFT(xor, int32_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT(xor, int64_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT(xor, uint32_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT(xor, uint64_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT(xor, int32_t, int64_t);
REGISTE_INDIRECT_ATOMIC_SOFT(xor, int64_t, int64_t);
REGISTE_INDIRECT_ATOMIC_SOFT(xor, uint32_t, int64_t);
REGISTE_INDIRECT_ATOMIC_SOFT(xor, uint64_t, int64_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(xor, int32_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(xor, int64_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(xor, uint32_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(xor, uint64_t, int32_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(xor, int32_t, int64_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(xor, int64_t, int64_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(xor, uint32_t, int64_t);
REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK(xor, uint64_t, int64_t);
}

#undef REGISTE_INDIRECT_ATOMIC_SOFT
#undef REGISTE_INDIRECT_ATOMIC_SOFT_NO_MASK

#endif // defined(__DAV_C310__)
