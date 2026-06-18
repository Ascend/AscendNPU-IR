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

 #include "RegBase/Cumulative/Cumulative.h"
 #include "RegBase/VecUtils.h"
 #include "Utils.h"
 #include "Vector/Transpose/TransposeUtils.h"
 #include "Vector/VecUtils.h"
 
 #if defined(__DAV_C310__)
 
 template <typename T>
 __aiv__ __attribute__((always_inline)) void
 sklansky_calculation(memref_t<__ubuf__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst,
                      sklansky_param_t param) {
   int32_t groupTailCount = param.groupTailCount;
   uint16_t groupTailCountu16 = (uint16_t)groupTailCount;
   int32_t addTailCount = param.addTailCount;
   uint16_t addTailCountu16 = (uint16_t)addTailCount;
   uint16_t addCount = (uint16_t)param.addCount;
   int32_t startOffset = param.startOffset;
   int32_t group_offset = param.group_offset;
   uint16_t group_offsetu16 = (uint16_t)group_offset;
   int32_t groupMainCount = param.groupMainCount;
   uint16_t groupMainCountu16 = (uint16_t)groupMainCount;
   int32_t mFold = param.mFactor;
   int32_t nAddFactor = param.nAddFactor;
   int32_t realDupSize = param.realDupSize;
   int32_t flag = param.flag;
   constexpr uint16_t num_per_reg = REG_REGISTER_SIZE / sizeof(T);
   __ubuf__ T *src_ptr = dst->aligned + dst->offset;
   __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
   uint16_t nLoop = CEIL_DIV(realDupSize, REG_REGISTER_SIZE);
   uint16_t src_stride0 = (uint16_t)src->strides[0];
   __VEC_SCOPE__ {
     VectorReg<T> x1RegTensor, x2RegTensor;
     vector_bool mask;
     uint32_t totalElements = (uint32_t)nAddFactor * (uint32_t)mFold;
     for (uint16_t j = 0; j < nLoop; j++) {
       CREATE_MASK_BY_SIZE(mask, T, totalElements);
       for (uint16_t m = 0; m < groupMainCountu16; m++) {
         int32_t src_offset = flag * m * group_offset + j * num_per_reg;
         vlds(x1RegTensor, src_ptr + startOffset, src_offset, NORM);
         for (uint16_t n = 1; n <= addCount; n++) {
           int32_t dst_offset =
               flag * (m * group_offset + n * src_stride0) + j * num_per_reg;
           vlds(x2RegTensor, src_ptr + startOffset, dst_offset, NORM);
           vadd(x2RegTensor, x1RegTensor, x2RegTensor, mask);
           vsts(x2RegTensor, dst_ptr + startOffset, dst_offset, NORM_B32, mask);
         }
       }
 
       for (uint16_t m = 0; m < groupTailCountu16; m++) {
         int32_t src_offset =
             flag * groupMainCount * group_offset + j * num_per_reg;
         vlds(x1RegTensor, src_ptr + startOffset, src_offset, NORM);
         for (uint16_t n = 1; n <= addTailCountu16; n++) {
           int32_t dst_offset =
               flag * (groupMainCount * group_offset + n * src_stride0) +
               j * num_per_reg;
           vlds(x2RegTensor, src_ptr + startOffset, dst_offset, NORM);
           vadd(x2RegTensor, x1RegTensor, x2RegTensor, mask, MODE_ZEROING);
           vsts(x2RegTensor, dst_ptr + startOffset, dst_offset, NORM_B32, mask);
         }
       }
     }
   }
 }
 
 template <typename T>
 __aiv__ __attribute__((always_inline)) void
 sklansky_loops(memref_t<__ubuf__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst,
                int32_t addLoop, int32_t rFactor, sklansky_param_t param) {
   int32_t addLoopPow = pow(2, addLoop); // 2^k：源距组起点的行距,
   int32_t nextAddLoopPow = pow(2, addLoop + 1); // 2^(k+1)：组大小
   // 源行在组内的位置（正向：第 2^k-1 行，反向：倒数第 2^k 行）
   param.startOffset = (param.flag == -1)
                           ? (rFactor - addLoopPow) * src->strides[0]
                           : (addLoopPow - 1) * src->strides[0];
   param.addCount =
       addLoopPow; // 每源加 2^k 个目标, 每个源行需要加的目标行数 = 2^k
   param.group_offset = nextAddLoopPow * src->strides[0]; // 组间距（字节）
   param.groupMainCount = rFactor / nextAddLoopPow;       // 完整组数
   param.groupTailCount =
       (rFactor % nextAddLoopPow > addLoopPow) ? 1 : 0; // 是否有尾组（0 或 1）
   param.addTailCount = rFactor - param.groupMainCount * nextAddLoopPow -
                        addLoopPow; // 尾组内目标行数
   sklansky_calculation(src, dst, param);
 }
 
 template <typename T>
 __aiv__ __attribute__((always_inline)) void
 oneway_sklansky_cumsum(memref_t<__ubuf__ T, 3> *src,
                        memref_t<__ubuf__ T, 3> *dst, bool reverse) {
   int32_t rFactor = src->sizes[0];
   int32_t mFactor = src->sizes[1];
   int32_t nFactor = src->sizes[2];
   int32_t dupSize_ = CEIL_FACTOR(nFactor * sizeof(T), 32);
   int32_t nAddFactor = dupSize_ / sizeof(T);
   int32_t realDupSize = nAddFactor * sizeof(T) * mFactor;
   int32_t rLoop = CeilLog2(rFactor); // 迭代轮数 = log2(rFactor_)
   sklansky_param_t param;
   param.flag = reverse ? -1 : 1;
   param.mFactor = mFactor;
   param.realDupSize = realDupSize;
   param.nAddFactor = nAddFactor;
   for (int32_t k = 0; k < rLoop; k++) {
     sklansky_loops(src, dst, k, rFactor, param);
   }
 }
 
 template <typename T>
 __aiv__ __attribute__((always_inline)) void
 oneway_cumsum_compensated(memref_t<__ubuf__ T, 3> *src,
                           memref_t<__ubuf__ T, 3> *dst, bool reverse) {
   int32_t rFactor = src->sizes[0];
   int32_t mFactor = src->sizes[1];
   int32_t nFactor = src->sizes[2];
   int32_t dupSize_ = CEIL_FACTOR(nFactor * sizeof(T), 32);
   int32_t nAddFactor = dupSize_ / sizeof(T);
   int32_t realDupSize = nAddFactor * sizeof(T) * mFactor;
   uint16_t nLoop = CEIL_DIV(realDupSize, REG_REGISTER_SIZE);
   uint16_t rLoop = (uint16_t)rFactor;
   int32_t num_per_reg = REG_REGISTER_SIZE / sizeof(T);
   int32_t stride0 = src->strides[0];
   __ubuf__ T *src_ptr = src->aligned + src->offset;
   __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
   int32_t start_row_offset = reverse ? (rFactor - 1) * stride0 : 0;
   __VEC_SCOPE__ {
     VectorReg<T> hi, lo, x2, s, bb, t1, t2;
     vector_bool mask;
     uint32_t totalElements = (uint32_t)nAddFactor * (uint32_t)mFactor;
     for (uint16_t j = 0; j < nLoop; j++) {
       CREATE_MASK_BY_SIZE(mask, T, totalElements);
       // row 0: (hi, lo) = (X[row0], 0)
       vlds(hi, src_ptr, start_row_offset + j * num_per_reg, NORM);
       vsub(lo, hi, hi, mask);                 // lo = 0
       vsts(hi, dst_ptr, start_row_offset + j * num_per_reg, NORM_B32, mask);
       for (uint16_t r = 1; r < rLoop; r++) {
         int32_t r_offset = reverse ? (rFactor - 1 - r) : r;
         vlds(x2, src_ptr + r_offset * stride0, j * num_per_reg, NORM);
         // TwoSum(hi, x2): s = fl(hi + x2), error e (x2 has no low part)
         vadd(s, hi, x2, mask);                // s = hi + x2
         vsub(bb, s, hi, mask);                // bb = s - hi
         vsub(t1, s, bb, mask);                // t1 = s - bb
         vsub(t1, hi, t1, mask);               // t1 = hi - (s - bb)
         vsub(t2, x2, bb, mask);               // t2 = x2 - bb
         vadd(t1, t1, t2, mask);               // t1 = e  (TwoSum error)
         vadd(t1, t1, lo, mask);               // t1 = e + lo  (fold old comp)
         // FastTwoSum(s, e) -> renormalized (hi, lo)
         vadd(hi, s, t1, mask);                // hi = s + e
         vsub(t2, hi, s, mask);                // t2 = hi - s
         vsub(lo, t1, t2, mask);               // lo = e - (hi - s)
         vsts(hi, dst_ptr + r_offset * stride0, j * num_per_reg, NORM_B32, mask);
       }
     }
   }
 }
 
 
 template <typename T, int dim>
 __aiv__ __attribute__((always_inline)) void
 compute_cumsum_3d(memref_t<__ubuf__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst,
                   bool reverse, memref_t<__ubuf__ T, 3> *temp = nullptr) {
   if constexpr (dim == 0) {
     copy_for_cum_op(src, dst);
     oneway_cumsum_compensated<T>(dst, dst, reverse);
   } else if constexpr (dim == 1) {
     int64_t n_aligned = CEIL_FACTOR(temp->sizes[2] * sizeof(T), 32) / sizeof(T);
     memref_t<__ubuf__ T, 3> temp_3d{
         temp->allocated,
         temp->aligned,
         temp->offset,
         {temp->sizes[1], temp->sizes[0], temp->sizes[2]},
         {temp->sizes[0] * n_aligned, n_aligned, 1}};
     transpose_dim_01<T>(src, &temp_3d);
     oneway_cumsum_compensated<T>(&temp_3d, &temp_3d, reverse);
     transpose_dim_01<T>(&temp_3d, dst);
   }
 }
 
 template <typename T>
 __aiv__ __attribute__((always_inline)) void
 compute_cumsum_2d_along_dim1(memref_t<__ubuf__ T, 2> *src,
                              memref_t<__ubuf__ T, 2> *dst, bool reverse) {
   int64_t M = src->sizes[0];
   uint16_t N = src->sizes[1];
   int32_t stride0 = src->strides[0];
   uint32_t num_per_reg = REG_REGISTER_SIZE / sizeof(T);
   using IdxT = std::conditional_t<sizeof(T) == 4, int32_t, int16_t>;
   using IdxUT = std::conditional_t<sizeof(T) == 4, uint32_t, uint16_t>;
   __ubuf__ T *src_ptr = src->aligned + src->offset;
   __ubuf__ T *dst_ptr = dst->aligned + dst->offset;
   uint16_t mLoop = CEIL_DIV((uint16_t)M, (uint16_t)num_per_reg);
   uint32_t size0 = M;
   int32_t start_col = reverse ? (N - 1) : 0;
   __VEC_SCOPE__ {
     VectorReg<T> x1RegTensor, x2RegTensor;
     VectorReg<IdxT> idx_reg;
     vector_bool mask;
     uint32_t full_mask_size = num_per_reg;
     CREATE_MASK_BY_SIZE(mask, T, full_mask_size);
     vci(idx_reg, 0);
     vmuls(idx_reg, idx_reg, stride0, mask);
     for (uint16_t m = 0; m < mLoop; m++) {
       __ubuf__ T *src_block = src_ptr + (int32_t)m * num_per_reg * stride0;
       __ubuf__ T *dst_block = dst_ptr + (int32_t)m * num_per_reg * stride0;
       CREATE_MASK_BY_SIZE(mask, T, size0);
       vgather2(x1RegTensor, src_block + start_col, (VectorReg<IdxUT> &)idx_reg,
                mask);
       vscatter(x1RegTensor, dst_block + start_col, (VectorReg<IdxUT> &)idx_reg,
                mask);
       for (uint16_t n = 1; n < N; n++) {
         int32_t nOffset = reverse ? (N - n - 1) : n;
         vgather2(x2RegTensor, src_block + nOffset, (VectorReg<IdxUT> &)idx_reg,
                  mask);
         vadd(x1RegTensor, x1RegTensor, x2RegTensor, mask);
         vscatter(x1RegTensor, dst_block + nOffset, (VectorReg<IdxUT> &)idx_reg,
                  mask);
       }
     }
   }
 }
 
 template <typename T, int cum_dim>
 __aiv__ __attribute__((always_inline)) void
 vector_cumsum_3d(memref_t<__ubuf__ T, 3> *src, memref_t<__ubuf__ T, 3> *dst,
                  memref_t<__ubuf__ T, 3> *temp, bool reverse) {
   static_assert(
       std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value ||
           std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value ||
           std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value ||
           std::is_same<T, float>::value || std::is_same<T, half>::value ||
           std::is_same<T, bfloat16_t>::value,
       "cumsum op only uint8/16/32_t, int8/16/32_t, float16/32 and  bfloat16 "
       "type operands in template!");
   compute_cumsum_3d<T, cum_dim>(src, dst, reverse, temp);
 }
 
 template <typename T, int cum_dim>
 __aiv__ __attribute__((always_inline)) void
 vector_cumsum_2d(memref_t<__ubuf__ T, 2> *src, memref_t<__ubuf__ T, 2> *dst,
                  memref_t<__ubuf__ T, 2> *temp, bool reverse) {
   static_assert(
       std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value ||
           std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value ||
           std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value ||
           std::is_same<T, float>::value || std::is_same<T, half>::value ||
           std::is_same<T, bfloat16_t>::value,
       "cumsum ra op only uint8/16/32/64_t, int8/16/32/64_t, float16/32 and  "
       "bfloat16 type operands in template!");
 
   if constexpr (cum_dim == 0) {
     memref_t<__ubuf__ T, 3> src_3d{
         src->allocated,
         src->aligned,
         src->offset,
         {src->sizes[0], 1, src->sizes[1]},
         {src->strides[0], src->strides[0], src->strides[1]}};
     memref_t<__ubuf__ T, 3> dst_3d{
         dst->allocated,
         dst->aligned,
         dst->offset,
         {dst->sizes[0], 1, dst->sizes[1]},
         {dst->strides[0], dst->strides[0], dst->strides[1]}};
     compute_cumsum_3d<T, 0>(&src_3d, &dst_3d, reverse);
   } else {
     if constexpr (sizeof(T) == 1) {
       memref_t<__ubuf__ T, 2> temp_trans{temp->allocated,
                                          temp->aligned,
                                          temp->offset,
                                          {src->sizes[1], src->sizes[0]},
                                          {temp->strides[0], temp->strides[1]}};
       transpose_ar2ra_i8<T>(src, &temp_trans);
       vector_cumsum_2d<T, 0>(&temp_trans, &temp_trans, nullptr, reverse);
       transpose_ar2ra_i8<T>(&temp_trans, dst);
     } else {
       memref_t<__ubuf__ T, 2> temp_trans{temp->allocated,
                                          temp->aligned,
                                          temp->offset,
                                          {src->sizes[1], src->sizes[0]},
                                          {temp->strides[0], temp->strides[1]}};
       transpose_ar2ra<T>(src, &temp_trans);
       vector_cumsum_2d<T, 0>(&temp_trans, &temp_trans, nullptr, reverse);
       transpose_ar2ra<T>(&temp_trans, dst);
     }
   }
 }
 
 template <typename T, typename IdxT, typename uIdxT>
 __aiv__ __attribute__((always_inline)) void
 copy_lane_gather(__ubuf__ T *s, __ubuf__ T *d, int n) {
   uint32_t nn = static_cast<uint32_t>(n);
   __VEC_SCOPE__ {
     VectorReg<T> a;
     VectorReg<IdxT> idx;
     vector_bool m;
     CREATE_MASK_BY_SIZE(m, T, nn);
     vci(idx, 0);
     vgather2(a, s, (VectorReg<uIdxT> &)idx, m);
     vscatter(a, d, (VectorReg<uIdxT> &)idx, m);
   }
 }
 
 template <typename T>
 __aiv__ __attribute__((always_inline)) void
 unified_scan_fwd_nocopy(__ubuf__ T *srcBase, __ubuf__ T *dstBase, int N,
                         int BpE) {
   using IdxT = std::conditional_t<sizeof(T) == 2, int16_t, int32_t>;
   using uIdxT = std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>;
   int numBlocks = (N + BpE - 1) / BpE;
   int tailSize = N - (numBlocks - 1) * BpE;
   int fullRegs = (tailSize < BpE) ? (numBlocks - 1) : numBlocks;
   int last = numBlocks - 1;
   for (int d = 1; d < N; d <<= 1) {
     __ubuf__ T *rbase = (d == 1) ? srcBase : dstBase; // read src on the 1st step
     __ubuf__ T *wbase = dstBase;
     int cf = (d + BpE - 1) / BpE; // first full register with rStart >= d
     bool tailG = (tailSize < BpE) && (last * BpE + tailSize > d);
     bool strG = (cf >= 1) && (cf * BpE > d) && (cf - 1 < fullRegs);
     int tailKeep = (d > last * BpE) ? (d - last * BpE) : 0;
     __ubuf__ T *tailRd = rbase + last * BpE + tailKeep;
     __ubuf__ T *tailWd = wbase + last * BpE + tailKeep;
     uint32_t tailN = static_cast<uint32_t>(tailSize - tailKeep);
     __ubuf__ T *strRd = rbase + d; // straddle: first active elem is global index d
     __ubuf__ T *strWd = wbase + d;
     uint32_t strN = static_cast<uint32_t>(cf * BpE - d); // BpE-(d-(cf-1)*BpE)
     uint32_t bpeU = static_cast<uint32_t>(BpE);
     // contiguous full registers [cf, fullRegs-1] processed HIGH->LOW; a vector-loop
     // induction var must be an ascending uint16_t, so iterate k and map k -> r.
     int fwdCnt = (fullRegs - cf > 0) ? (fullRegs - cf) : 0;
     uint16_t fwdCntU = static_cast<uint16_t>(fwdCnt);
     __VEC_SCOPE__ {
       VectorReg<T> a, b;
       VectorReg<IdxT> idx;
       vector_align valign;
       vector_bool m, mFull;
       vci(idx, 0);
       CREATE_MASK_BY_SIZE(mFull, T, bpeU);
       if (tailG) { // highest active register
         CREATE_MASK_BY_SIZE(m, T, tailN);
         vgather2(a, tailRd, (VectorReg<uIdxT> &)idx, m);
         vgather2(b, tailRd - d, (VectorReg<uIdxT> &)idx, m);
         vadd(a, a, b, m);
         vscatter(a, tailWd, (VectorReg<uIdxT> &)idx, m);
       }
       for (uint16_t k = 0; k < fwdCntU; ++k) { // contiguous bulk, high->low
         int r = (fullRegs - 1) - static_cast<int>(k);
         __ubuf__ T *rb = rbase + r * BpE;
         __ubuf__ T *wb = wbase + r * BpE;
         vlds(a, rb, 0, NORM);
         vldas(valign, rb - d);
         vldus(b, valign, rb - d);
         vadd(a, a, b, mFull);
         vsts(a, wb, 0, NORM_B32, mFull);
       }
       if (strG) { // lowest active register
         CREATE_MASK_BY_SIZE(m, T, strN);
         vgather2(a, strRd, (VectorReg<uIdxT> &)idx, m);
         vgather2(b, strRd - d, (VectorReg<uIdxT> &)idx, m);
         vadd(a, a, b, m);
         vscatter(a, strWd, (VectorReg<uIdxT> &)idx, m);
       }
     }
     if (d == 1) {
       copy_lane_gather<T, IdxT, uIdxT>(srcBase, dstBase, d);
     }
   }
 }
 
 template <typename T>
 __aiv__ __attribute__((always_inline)) void
 unified_scan_rev_nocopy(__ubuf__ T *srcBase, __ubuf__ T *dstBase, int N,
                         int BpE) {
   using IdxT = std::conditional_t<sizeof(T) == 2, int16_t, int32_t>;
   using uIdxT = std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>;
   int numBlocks = (N + BpE - 1) / BpE;
   int tailSize = N - (numBlocks - 1) * BpE;
   int fullRegs = (tailSize < BpE) ? (numBlocks - 1) : numBlocks;
   int last = numBlocks - 1;
   for (int d = 1; d < N; d <<= 1) {
     __ubuf__ T *rbase = (d == 1) ? srcBase : dstBase; // read src on the 1st step
     __ubuf__ T *wbase = dstBase;
     int dq = d / BpE;
     int rFA = (N - d >= BpE) ? (N - d - BpE) / BpE : -1; // last fully active
     int rSF = fullRegs - 2 - dq;                         // last src-in-full
     int rCHi = (rFA < rSF) ? rFA : rSF;                  // last contiguous reg
     int lastActive = (N - d - 1) / BpE;                  // highest active reg
     int gLo = (rCHi + 1 > 0) ? (rCHi + 1) : 0;
     uint32_t bpeU = static_cast<uint32_t>(BpE);
     // vector-loop induction vars must be ascending uint16_t; iterate k and map to r.
     int revCCnt = (rCHi >= 0) ? (rCHi + 1) : 0;          // # contiguous regs
     uint16_t revCCntU = static_cast<uint16_t>(revCCnt);
     int revGCnt = (lastActive - gLo + 1 > 0) ? (lastActive - gLo + 1) : 0;
     uint16_t revGCntU = static_cast<uint16_t>(revGCnt);
     __VEC_SCOPE__ {
       VectorReg<T> a, b;
       VectorReg<IdxT> idx;
       vector_align valign;
       vector_bool m, mFull;
       vci(idx, 0);
       CREATE_MASK_BY_SIZE(mFull, T, bpeU);
       for (uint16_t k = 0; k < revCCntU; ++k) { // contiguous bulk, low->high
         int r = static_cast<int>(k);
         __ubuf__ T *rb = rbase + r * BpE;
         __ubuf__ T *wb = wbase + r * BpE;
         vlds(a, rb, 0, NORM);
         vldas(valign, rb + d);
         vldus(b, valign, rb + d);
         vadd(a, a, b, mFull);
         vsts(a, wb, 0, NORM_B32, mFull);
       }
       for (uint16_t k = 0; k < revGCntU; ++k) { // gather boundary / rest
         int r = gLo + static_cast<int>(k);
         int rStart = r * BpE;
         int sz = (r == last) ? tailSize : BpE;
         int hi = rStart + sz;
         int addEnd = (hi < (N - d)) ? hi : (N - d);
         uint32_t nact = static_cast<uint32_t>(addEnd - rStart);
         CREATE_MASK_BY_SIZE(m, T, nact);
         __ubuf__ T *rd = rbase + rStart;
         __ubuf__ T *wd = wbase + rStart;
         vgather2(a, rd, (VectorReg<uIdxT> &)idx, m);
         vgather2(b, rd + d, (VectorReg<uIdxT> &)idx, m);
         vadd(a, a, b, m);
         vscatter(a, wd, (VectorReg<uIdxT> &)idx, m);
       }
     }
     if (d == 1){
       copy_lane_gather<T, IdxT, uIdxT>(srcBase + (N - d), dstBase + (N - d), d);
     }
   }
 }
 
 template <typename T>
 __aiv__ __attribute__((always_inline)) void
 unified_scan_gather_nocopy(__ubuf__ T *srcBase, __ubuf__ T *dstBase, int N,
                            int BpE, bool reverse) {
   if (reverse)
     unified_scan_rev_nocopy<T>(srcBase, dstBase, N, BpE);
   else
     unified_scan_fwd_nocopy<T>(srcBase, dstBase, N, BpE);
 }
 
 template <typename T>
 __aiv__ __attribute__((always_inline)) T
 scalar_incl_prefix_sum_4way(__ubuf__ T *src, __ubuf__ T *dst, int32_t N, T seed) {
   T acc = seed;
   int32_t i = 0;
   int32_t imain = N - (N & 3);
   for (; i < imain; i += 4) {
     T T0 = src[i + 0];
     T T1 = src[i + 1];
     T T2 = src[i + 2];
     T T3 = src[i + 3];
     T s01   = (T)(T0 + T1);
     T s012  = (T)(s01 + T2);
     T s0123 = (T)(s012 + T3);
     dst[i + 0] = (T)(acc + T0);
     dst[i + 1] = (T)(acc + s01);
     dst[i + 2] = (T)(acc + s012);
     dst[i + 3] = (T)(acc + s0123);
     acc = (T)(acc + s0123);
   }
   for (; i < N; i++) {
     acc = (T)(acc + src[i]);
     dst[i] = acc;
   }
   return acc;
 }
 
 template <typename T>
 __aiv__ __attribute__((always_inline)) T
 scalar_incl_suffix_sum_4way(__ubuf__ T *src, __ubuf__ T *dst, int32_t N, T seed) {
   T acc = seed;
   int32_t i = N - 1;
   for (; i >= 3; i -= 4) {
     T T0 = src[i];          // highest index in this group of 4
     T T1 = src[i - 1];
     T T2 = src[i - 2];
     T T3 = src[i - 3];      // lowest index in this group
     T s01   = (T)(T0 + T1);
     T s012  = (T)(s01 + T2);
     T s0123 = (T)(s012 + T3);
     dst[i]     = (T)(acc + T0);
     dst[i - 1] = (T)(acc + s01);
     dst[i - 2] = (T)(acc + s012);
     dst[i - 3] = (T)(acc + s0123);
     acc = (T)(acc + s0123);
   }
   for (; i >= 0; i--) {
     acc = (T)(acc + src[i]);
     dst[i] = acc;
   }
   return acc;
 }
 
 template <typename PromoteDataType>
 __aiv__ __attribute__((always_inline)) void
 vector_cumsum_1d_twoway(memref_t<__ubuf__ PromoteDataType, 1> *src,
                         memref_t<__ubuf__ PromoteDataType, 1> *dst,
                         bool reverse) {
 
   constexpr int BpE = REG_REGISTER_SIZE / sizeof(PromoteDataType);
   int64_t N = dst->sizes[0];
   int numBlocks = static_cast<int>(CEIL_DIV(N, static_cast<int64_t>(BpE)));
   INTRINSIC(pipe_barrier, PIPE_ALL);
   if (N <= 2 * BpE || sizeof(PromoteDataType) < 2) {
     if (!reverse) {
       (void)scalar_incl_prefix_sum_4way<PromoteDataType>(src->aligned + src->offset, dst->aligned + dst->offset,
                                                 (int32_t)N, (PromoteDataType)0);
     }
     else {
       (void)scalar_incl_suffix_sum_4way<PromoteDataType>(src->aligned + src->offset, dst->aligned + dst->offset,
                                                 (int32_t)N, (PromoteDataType)0);
     }
     INTRINSIC(pipe_barrier, PIPE_ALL);
     return;
   }
   if constexpr (sizeof(PromoteDataType) >= 2) {
     __ubuf__ PromoteDataType *dpBase = dst->aligned + dst->offset;
     unified_scan_gather_nocopy<PromoteDataType>(src->aligned + src->offset, dst->aligned + dst->offset,
                                              static_cast<int>(N), BpE, reverse);
     INTRINSIC(pipe_barrier, PIPE_ALL);
     return;
   }
   return;
 }
 
 template <typename T, int cum_dim>
 __aiv__ __attribute__((always_inline)) void
 vector_cumsum_1d(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ T, 1> *dst,
                  memref_t<__ubuf__ T, 1> *temp, bool reverse) {
   static_assert(
       std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value ||
           std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value ||
           std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value ||
           std::is_same<T, float>::value || std::is_same<T, half>::value ||
           std::is_same<T, bfloat16_t>::value,
       "cumsum op only uint8/16/32/64_t, int8/16/32/64_t, float16/32 and "
       "bfloat16 type operands in template!");
   vector_cumsum_1d_twoway<T>(src, dst, reverse);
 }
 extern "C" {
 
 REGISTER_CUMSUM(2, int8_t, 0);
 REGISTER_CUMSUM(2, uint8_t, 0);
 REGISTER_CUMSUM(2, int16_t, 0);
 REGISTER_CUMSUM(2, uint16_t, 0);
 REGISTER_CUMSUM(2, int32_t, 0);
 REGISTER_CUMSUM(2, uint32_t, 0);
 REGISTER_CUMSUM(2, half, 0);
 REGISTER_CUMSUM(2, float, 0);
 REGISTER_CUMSUM(2, bfloat16_t, 0);
 
 REGISTER_CUMSUM_WITH_TEMP(2, int8_t, 1);
 REGISTER_CUMSUM_WITH_TEMP(2, uint8_t, 1);
 REGISTER_CUMSUM_WITH_TEMP(2, int16_t, 1);
 REGISTER_CUMSUM_WITH_TEMP(2, uint16_t, 1);
 REGISTER_CUMSUM_WITH_TEMP(2, int32_t, 1);
 REGISTER_CUMSUM_WITH_TEMP(2, uint32_t, 1);
 REGISTER_CUMSUM_WITH_TEMP(2, half, 1);
 REGISTER_CUMSUM_WITH_TEMP(2, float, 1);
 REGISTER_CUMSUM_WITH_TEMP(2, bfloat16_t, 1);
 
 REGISTER_CUMSUM(3, int8_t, 0);
 REGISTER_CUMSUM(3, uint8_t, 0);
 REGISTER_CUMSUM(3, int16_t, 0);
 REGISTER_CUMSUM(3, uint16_t, 0);
 REGISTER_CUMSUM(3, int32_t, 0);
 REGISTER_CUMSUM(3, uint32_t, 0);
 REGISTER_CUMSUM(3, half, 0);
 REGISTER_CUMSUM(3, float, 0);
 REGISTER_CUMSUM(3, bfloat16_t, 0);
 
 REGISTER_CUMSUM_WITH_TEMP(3, int8_t, 1);
 REGISTER_CUMSUM_WITH_TEMP(3, uint8_t, 1);
 REGISTER_CUMSUM_WITH_TEMP(3, int16_t, 1);
 REGISTER_CUMSUM_WITH_TEMP(3, uint16_t, 1);
 REGISTER_CUMSUM_WITH_TEMP(3, int32_t, 1);
 REGISTER_CUMSUM_WITH_TEMP(3, uint32_t, 1);
 REGISTER_CUMSUM_WITH_TEMP(3, half, 1);
 REGISTER_CUMSUM_WITH_TEMP(3, float, 1);
 REGISTER_CUMSUM_WITH_TEMP(3, bfloat16_t, 1);
 }
 
 #endif
