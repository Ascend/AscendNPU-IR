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

 #if defined(__DAV_C310__)

 #include "__clang_cce_simt_intrinsics.h"
 #include "RegBase/VecUtils.h"
 
 constexpr unsigned int MAX_THREAD_NUM = 1024;
 
 enum class IndirectAtomicOp { Add, Min, Max, Or, And, XOr, Xchg, Cas };
 
 template <typename DTYPE>
 inline constexpr bool kAtomicHalfPrecVoidReturn =
     std::is_same<DTYPE, half>::value ||
     std::is_same<DTYPE, bfloat16_t>::value;
 
 template <IndirectAtomicOp Op, typename DTYPE>
 __simt_callee__ __aiv__ __attribute__((always_inline)) static void
 indirectAtomicInvokeVoid(__gm__ DTYPE *addr, DTYPE val) {
   if constexpr (Op == IndirectAtomicOp::Add) {
     atomicAdd(addr, val);
   } else if constexpr (Op == IndirectAtomicOp::Min) {
     atomicMin(addr, val);
   } else if constexpr (Op == IndirectAtomicOp::Max) {
     atomicMax(addr, val);
   } else if constexpr (Op == IndirectAtomicOp::Or) {
     atomicOr(addr, val);
   } else if constexpr (Op == IndirectAtomicOp::And) {
     atomicAnd(addr, val);
   } else if constexpr (Op == IndirectAtomicOp::XOr) {
     atomicXOr(addr, val);
   } else if constexpr (Op == IndirectAtomicOp::Xchg) {
     atomicExch(addr, val);
   }
 }
 
 template <IndirectAtomicOp Op, typename DTYPE>
 __simt_callee__ __aiv__ __attribute__((always_inline)) static DTYPE
 indirectAtomicInvoke(__gm__ DTYPE *addr, DTYPE val) {
   if constexpr (Op == IndirectAtomicOp::Add) {
     return atomicAdd(addr, val);
   } else if constexpr (Op == IndirectAtomicOp::Min) {
     return atomicMin(addr, val);
   } else if constexpr (Op == IndirectAtomicOp::Max) {
     return atomicMax(addr, val);
   } else if constexpr (Op == IndirectAtomicOp::Or) {
     return atomicOr(addr, val);
   } else if constexpr (Op == IndirectAtomicOp::And) {
     return atomicAnd(addr, val);
   } else if constexpr (Op == IndirectAtomicOp::XOr) {
     return atomicXOr(addr, val);
   } else if constexpr (Op == IndirectAtomicOp::Xchg) {
     return atomicExch(addr, val);
   }
 }
 
 template <IndirectAtomicOp Op, typename DTYPE, typename ITYPE>
 __simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
 __aiv__ __attribute__((always_inline)) static void SimtIndirectAtomic(
     __gm__ DTYPE *src, __ubuf__ ITYPE *offsets, __ubuf__ DTYPE *value,
     __ubuf__ int8_t *mask, __ubuf__ DTYPE *out, const int64_t indicesSize,
     const int64_t offsetStride, const int64_t valueStride, const int64_t maskStride, 
     const int64_t outStride) {
   for (uint64_t idx = threadIdx.x; idx < indicesSize; idx += blockDim.x) {
     if (mask[idx * maskStride]) {
       ITYPE ldIdx = offsets[idx * offsetStride];
       if constexpr (kAtomicHalfPrecVoidReturn<DTYPE>) {
         DTYPE oldValue = *(src + ldIdx);
         indirectAtomicInvokeVoid<Op, DTYPE>(src + ldIdx,
                                             value[idx * valueStride]);
         out[idx * outStride] = oldValue;
       } else {
         DTYPE oldValue = indirectAtomicInvoke<Op, DTYPE>(
             src + ldIdx, value[idx * valueStride]);
         out[idx * outStride] = oldValue;
       }
     } else {
       out[idx * outStride] = MakeSentinelNegOne<DTYPE>();
     }
   }
 }
 
 template <IndirectAtomicOp Op, typename DTYPE, typename ITYPE>
 __simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
 __aiv__ __attribute__((always_inline)) static void SimtIndirectAtomicWithoutMask(
     __gm__ DTYPE *src, __ubuf__ ITYPE *offsets, __ubuf__ DTYPE *value,
     __ubuf__ DTYPE *out, const int64_t indicesSize, const int64_t offsetStride,
     const int64_t valueStride, const int64_t outStride) {
   for (uint64_t idx = threadIdx.x; idx < indicesSize; idx += blockDim.x) {
     ITYPE ldIdx = offsets[idx * offsetStride];
     if constexpr (kAtomicHalfPrecVoidReturn<DTYPE>) {
       DTYPE oldValue = *(src + ldIdx);
       indirectAtomicInvokeVoid<Op, DTYPE>(src + ldIdx,
                                           value[idx * valueStride]);
       out[idx * outStride] = oldValue;
     } else {
       DTYPE oldValue = indirectAtomicInvoke<Op, DTYPE>(
           src + ldIdx, value[idx * valueStride]);
       out[idx * outStride] = oldValue;
     }
   }
 }
 
 template <typename DTYPE, typename ITYPE>
 __simt_vf__ LAUNCH_BOUND(MAX_THREAD_NUM)
 __aiv__ __attribute__((always_inline)) static void SimtIndirectAtomicCas(
     __gm__ DTYPE *src, __ubuf__ ITYPE *offsets, __ubuf__ DTYPE *comp,
     __ubuf__ DTYPE *value, __ubuf__ DTYPE *out, const int64_t indicesSize,
     const int64_t offsetStride, const int64_t compStride,
     const int64_t valueStride, const int64_t outStride) {
   for (uint64_t idx = threadIdx.x; idx < indicesSize; idx += blockDim.x) {
     ITYPE ldIdx = offsets[idx * offsetStride];
     DTYPE oldValue = atomicCAS(src + ldIdx, comp[idx * compStride],
                                value[idx * valueStride]);
     out[idx * outStride] = oldValue;
   }
 }
 
 template <typename T>
 inline constexpr bool kIsValidItype =
     std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>;

 template <IndirectAtomicOp Op, typename T>
 __aiv__ __attribute__((always_inline)) static constexpr bool
 isValidAtomicOpDtype() {
   if constexpr (Op == IndirectAtomicOp::Add ||
                 Op == IndirectAtomicOp::Min ||
                 Op == IndirectAtomicOp::Max) {
     return std::is_same_v<T, float> || std::is_same_v<T, half> ||
            std::is_same_v<T, half2> || std::is_same_v<T, bfloat16_t> ||
            std::is_same_v<T, bfloat16x2_t> || std::is_same_v<T, int32_t> ||
            std::is_same_v<T, int64_t> || std::is_same_v<T, uint32_t> ||
            std::is_same_v<T, uint64_t>;
   } else if constexpr (Op == IndirectAtomicOp::Or ||
                        Op == IndirectAtomicOp::And ||
                        Op == IndirectAtomicOp::XOr) {
     return std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> ||
            std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>;
   } else if constexpr (Op == IndirectAtomicOp::Xchg ||
                        Op == IndirectAtomicOp::Cas) {
     return std::is_same_v<T, float> || std::is_same_v<T, half2> ||
            std::is_same_v<T, bfloat16x2_t> || std::is_same_v<T, int32_t> ||
            std::is_same_v<T, int64_t> || std::is_same_v<T, uint32_t> ||
            std::is_same_v<T, uint64_t>;
   } else {
     return false;
   }
 }
 
 template <IndirectAtomicOp Op, typename DTYPE, typename ITYPE>
 __aiv__ __attribute__((always_inline)) static void indirect_atomic_impl(
     memref_t<__gm__ DTYPE, 1> *src, memref_t<__ubuf__ ITYPE, 1> *offsets,
     memref_t<__ubuf__ DTYPE, 1> *value, memref_t<__ubuf__ int8_t, 1> *mask,
     memref_t<__ubuf__ DTYPE, 1> *out) {
   cce::async_invoke<SimtIndirectAtomic<Op, DTYPE, ITYPE>>(
       cce::dim3{MAX_THREAD_NUM}, reinterpret_cast<__gm__ DTYPE *>(src->aligned),
       reinterpret_cast<__ubuf__ ITYPE *>(offsets->aligned + offsets->offset),
       reinterpret_cast<__ubuf__ DTYPE *>(value->aligned + value->offset),
       reinterpret_cast<__ubuf__ int8_t *>(mask->aligned + mask->offset),
       reinterpret_cast<__ubuf__ DTYPE *>(out->aligned + out->offset),
       offsets->sizes[0], offsets->strides[0], value->strides[0],
       mask->strides[0], out->strides[0]);
 }
 
 template <IndirectAtomicOp Op, typename DTYPE, typename ITYPE>
 __aiv__ __attribute__((always_inline)) static void indirect_atomic_no_mask_impl(
     memref_t<__gm__ DTYPE, 1> *src, memref_t<__ubuf__ ITYPE, 1> *offsets,
     memref_t<__ubuf__ DTYPE, 1> *value, memref_t<__ubuf__ DTYPE, 1> *out) {
   cce::async_invoke<SimtIndirectAtomicWithoutMask<Op, DTYPE, ITYPE>>(
       cce::dim3{MAX_THREAD_NUM}, reinterpret_cast<__gm__ DTYPE *>(src->aligned),
       reinterpret_cast<__ubuf__ ITYPE *>(offsets->aligned + offsets->offset),
       reinterpret_cast<__ubuf__ DTYPE *>(value->aligned + value->offset),
       reinterpret_cast<__ubuf__ DTYPE *>(out->aligned + out->offset),
       offsets->sizes[0], offsets->strides[0], value->strides[0],
       out->strides[0]);
 }
 
 template <typename DTYPE, typename ITYPE>
 __aiv__ __attribute__((always_inline)) static void indirect_atomic_cas_impl(
     memref_t<__gm__ DTYPE, 1> *src, memref_t<__ubuf__ ITYPE, 1> *offsets,
     memref_t<__ubuf__ DTYPE, 1> *comp, memref_t<__ubuf__ DTYPE, 1> *value,
     memref_t<__ubuf__ DTYPE, 1> *out) {
   cce::async_invoke<SimtIndirectAtomicCas<DTYPE, ITYPE>>(
       cce::dim3{MAX_THREAD_NUM}, reinterpret_cast<__gm__ DTYPE *>(src->aligned),
       reinterpret_cast<__ubuf__ ITYPE *>(offsets->aligned + offsets->offset),
       reinterpret_cast<__ubuf__ DTYPE *>(comp->aligned + comp->offset),
       reinterpret_cast<__ubuf__ DTYPE *>(value->aligned + value->offset),
       reinterpret_cast<__ubuf__ DTYPE *>(out->aligned + out->offset),
       offsets->sizes[0], offsets->strides[0], comp->strides[0],
       value->strides[0], out->strides[0]);
 }
 
 template <typename DTYPE, typename ITYPE,
           std::enable_if_t<isValidAtomicOpDtype<IndirectAtomicOp::Cas,
                                                 DTYPE>() &&
                                kIsValidItype<ITYPE>,
                            int> = 0>
 __aiv__ __attribute__((always_inline)) void indirect_atomic_cas(
     memref_t<__gm__ DTYPE, 1> *src, memref_t<__ubuf__ ITYPE, 1> *offsets,
     memref_t<__ubuf__ DTYPE, 1> *comp, memref_t<__ubuf__ DTYPE, 1> *value,
     memref_t<__ubuf__ DTYPE, 1> *out) {
   indirect_atomic_cas_impl<DTYPE, ITYPE>(src, offsets, comp, value, out);
 }
 
 #define DEFINE_INDIRECT_ATOMIC_WRAPPER(op, opEnum)                             \
  template <typename DTYPE, typename ITYPE,                                    \
            std::enable_if_t<                                                  \
                isValidAtomicOpDtype<IndirectAtomicOp::opEnum, DTYPE>() &&     \
                    kIsValidItype<ITYPE>,                                      \
                int> = 0>                                                      \
  __aiv__ __attribute__((always_inline)) void indirect_atomic_##op(            \
      memref_t<__gm__ DTYPE, 1> *src,                                          \
      memref_t<__ubuf__ ITYPE, 1> *offsets,                                    \
      memref_t<__ubuf__ DTYPE, 1> *value,                                      \
      memref_t<__ubuf__ int8_t, 1> *mask,                                      \
      memref_t<__ubuf__ DTYPE, 1> *out) {                                      \
    indirect_atomic_impl<IndirectAtomicOp::opEnum, DTYPE, ITYPE>(              \
        src, offsets, value, mask, out);                                       \
  }                                                                            \
  template <typename DTYPE, typename ITYPE,                                    \
            std::enable_if_t<                                                  \
                isValidAtomicOpDtype<IndirectAtomicOp::opEnum, DTYPE>() &&     \
                    kIsValidItype<ITYPE>,                                      \
                int> = 0>                                                      \
  __aiv__ __attribute__((always_inline))                                       \
  void indirect_atomic_##op##_no_mask(                                         \
      memref_t<__gm__ DTYPE, 1> *src,                                          \
      memref_t<__ubuf__ ITYPE, 1> *offsets,                                    \
      memref_t<__ubuf__ DTYPE, 1> *value,                                      \
      memref_t<__ubuf__ DTYPE, 1> *out) {                                      \
    indirect_atomic_no_mask_impl<IndirectAtomicOp::opEnum, DTYPE, ITYPE>(      \
        src, offsets, value, out);                                             \
  }

 DEFINE_INDIRECT_ATOMIC_WRAPPER(add,  Add)
 DEFINE_INDIRECT_ATOMIC_WRAPPER(min,  Min)
 DEFINE_INDIRECT_ATOMIC_WRAPPER(max,  Max)
 DEFINE_INDIRECT_ATOMIC_WRAPPER(or,   Or)
 DEFINE_INDIRECT_ATOMIC_WRAPPER(and,  And)
 DEFINE_INDIRECT_ATOMIC_WRAPPER(xor,  XOr)
 DEFINE_INDIRECT_ATOMIC_WRAPPER(xchg, Xchg)

 #undef DEFINE_INDIRECT_ATOMIC_WRAPPER
 
 extern "C" {
   REGISTE_INDIRECT_ATOMIC(add, float, int32_t);
   REGISTE_INDIRECT_ATOMIC(add, bfloat16_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(add, bfloat16x2_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(add, half2, int32_t);
   REGISTE_INDIRECT_ATOMIC(add, half, int32_t);
   REGISTE_INDIRECT_ATOMIC(add, int32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(add, int64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(add, uint32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(add, uint64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(add, float, int64_t);
   REGISTE_INDIRECT_ATOMIC(add, bfloat16_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(add, bfloat16x2_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(add, half2, int64_t);
   REGISTE_INDIRECT_ATOMIC(add, half, int64_t);
   REGISTE_INDIRECT_ATOMIC(add, int32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(add, int64_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(add, uint32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(add, uint64_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(add, float, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(add, bfloat16_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(add, bfloat16x2_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(add, half2, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(add, half, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(add, int32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(add, int64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(add, uint32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(add, uint64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(add, float, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(add, bfloat16_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(add, bfloat16x2_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(add, half2, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(add, half, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(add, int32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(add, int64_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(add, uint32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(add, uint64_t, int64_t);
 
   REGISTE_INDIRECT_ATOMIC(min, float, int32_t);
   REGISTE_INDIRECT_ATOMIC(min, bfloat16_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(min, bfloat16x2_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(min, half2, int32_t);
   REGISTE_INDIRECT_ATOMIC(min, half, int32_t);
   REGISTE_INDIRECT_ATOMIC(min, int32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(min, int64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(min, uint32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(min, uint64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(min, float, int64_t);
   REGISTE_INDIRECT_ATOMIC(min, bfloat16_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(min, bfloat16x2_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(min, half2, int64_t);
   REGISTE_INDIRECT_ATOMIC(min, half, int64_t);
   REGISTE_INDIRECT_ATOMIC(min, int32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(min, int64_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(min, uint32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(min, uint64_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(min, float, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(min, bfloat16_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(min, bfloat16x2_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(min, half2, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(min, half, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(min, int32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(min, int64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(min, uint32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(min, uint64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(min, float, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(min, bfloat16_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(min, bfloat16x2_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(min, half2, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(min, half, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(min, int32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(min, int64_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(min, uint32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(min, uint64_t, int64_t);
 
   REGISTE_INDIRECT_ATOMIC(max, float, int32_t);
   REGISTE_INDIRECT_ATOMIC(max, bfloat16_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(max, bfloat16x2_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(max, half2, int32_t);
   REGISTE_INDIRECT_ATOMIC(max, half, int32_t);
   REGISTE_INDIRECT_ATOMIC(max, int32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(max, int64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(max, uint32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(max, uint64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(max, float, int64_t);
   REGISTE_INDIRECT_ATOMIC(max, bfloat16_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(max, bfloat16x2_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(max, half2, int64_t);
   REGISTE_INDIRECT_ATOMIC(max, half, int64_t);
   REGISTE_INDIRECT_ATOMIC(max, int32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(max, int64_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(max, uint32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(max, uint64_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(max, float, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(max, bfloat16_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(max, bfloat16x2_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(max, half2, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(max, half, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(max, int32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(max, int64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(max, uint32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(max, uint64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(max, float, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(max, bfloat16_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(max, bfloat16x2_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(max, half2, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(max, half, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(max, int32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(max, int64_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(max, uint32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(max, uint64_t, int64_t);
 
   REGISTE_INDIRECT_ATOMIC(or, int32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(or, int64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(or, uint32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(or, uint64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(or, int32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(or, int64_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(or, uint32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(or, uint64_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(or, int32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(or, int64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(or, uint32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(or, uint64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(or, int32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(or, int64_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(or, uint32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(or, uint64_t, int64_t);
 
   REGISTE_INDIRECT_ATOMIC(and, int32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(and, int64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(and, uint32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(and, uint64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(and, int32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(and, int64_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(and, uint32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(and, uint64_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(and, int32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(and, int64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(and, uint32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(and, uint64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(and, int32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(and, int64_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(and, uint32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(and, uint64_t, int64_t);
 
   REGISTE_INDIRECT_ATOMIC(xor, int32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(xor, int64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(xor, uint32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(xor, uint64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(xor, int32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(xor, int64_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(xor, uint32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(xor, uint64_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(xor, int32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(xor, int64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(xor, uint32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(xor, uint64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(xor, int32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(xor, int64_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(xor, uint32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(xor, uint64_t, int64_t);
 
   REGISTE_INDIRECT_ATOMIC(xchg, float, int32_t);
   REGISTE_INDIRECT_ATOMIC(xchg, bfloat16x2_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(xchg, half2, int32_t);
   REGISTE_INDIRECT_ATOMIC(xchg, int32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(xchg, int64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(xchg, uint32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(xchg, uint64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC(xchg, float, int64_t);
   REGISTE_INDIRECT_ATOMIC(xchg, bfloat16x2_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(xchg, half2, int64_t);
   REGISTE_INDIRECT_ATOMIC(xchg, int32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(xchg, int64_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(xchg, uint32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC(xchg, uint64_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(xchg, float, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(xchg, bfloat16x2_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(xchg, half2, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(xchg, int32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(xchg, int64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(xchg, uint32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(xchg, uint64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(xchg, float, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(xchg, bfloat16x2_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(xchg, half2, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(xchg, int32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(xchg, int64_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(xchg, uint32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_NO_MASK(xchg, uint64_t, int64_t);
 
   REGISTE_INDIRECT_ATOMIC_CAS(float, int32_t);
   REGISTE_INDIRECT_ATOMIC_CAS(bfloat16x2_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_CAS(half2, int32_t);
   REGISTE_INDIRECT_ATOMIC_CAS(int32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_CAS(int64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_CAS(uint32_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_CAS(uint64_t, int32_t);
   REGISTE_INDIRECT_ATOMIC_CAS(float, int64_t);
   REGISTE_INDIRECT_ATOMIC_CAS(bfloat16x2_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_CAS(half2, int64_t);
   REGISTE_INDIRECT_ATOMIC_CAS(int32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_CAS(int64_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_CAS(uint32_t, int64_t);
   REGISTE_INDIRECT_ATOMIC_CAS(uint64_t, int64_t);
 }
 
#endif
