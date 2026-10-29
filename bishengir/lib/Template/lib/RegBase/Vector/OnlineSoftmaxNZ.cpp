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

// FP16 NZ16 scores -> FP8 NZ32 probabilities and FP32 block sums.
#include "ArchDefs.h"

#if BISHENG_TPL_C310
#include "RegBase/VecUtils.h"
#include "Utils.h"
#include <stdint.h>

// N512 FP16 ABI: half score/max in UB, FP32 exp/sum only in registers.
// P_offset(k,r,c) = 64*rows*k + 512*(r/16) + 32*(r%16) + c.
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_online_softmax_nz_f16_f8e4m3fn(
    memref_t<__ubuf__ half, 4> *scores, memref_t<__ubuf__ half, 1> *old_m,
    memref_t<__ubuf__ uint8_t, 1> *pack_index,
    memref_t<__ubuf__ float8_e4m3_t, 4> *p, memref_t<__ubuf__ half, 1> *new_m,
    memref_t<__ubuf__ float, 1> *block_sum) {
  auto s = scores->aligned + scores->offset;
  auto mi = old_m->aligned + old_m->offset;
  auto po = (__ubuf__ uint8_t *)(p->aligned + p->offset);
  auto mo = new_m->aligned + new_m->offset;
  auto so = block_sum->aligned + block_sum->offset;
  auto lut = pack_index->aligned + pack_index->offset;
  const uint16_t rows = scores->sizes[1] * 16;
  const uint32_t sn = scores->strides[0], sm = scores->strides[1];
  const uint32_t pn = p->strides[0], pm = p->strides[1];

  __VEC_SCOPE__ {
    vector_bool halves = pset_b16(PAT_ALL);
    vector_bool floats = pset_b32(PAT_ALL);
    vector_bool bytes = pset_b8(PAT_ALL);
    VectorReg<uint16_t> lane, row;
    vci((VectorReg<int16_t> &)lane, int16_t(0), INC_ORDER);
    vshrs(row, lane, int16_t(4), halves, MODE_ZEROING);
    VectorReg<uint8_t> indices;
    vlds(indices, lut, 0, NORM);

    VectorReg<half> x0_0, x1_0, x0_1, x1_1, x0_2, x1_2, x0_3, x1_3;
    VectorReg<half> xmax_0, xmax_1, xmax_2, xmax_3, previous, mn, packed_m;
    VectorReg<float> e0_0, e1_0, e2_0, e3_0, e0_1, e1_1, e2_1, e3_1, e0_2, e1_2,
        e2_2, e3_2, e0_3, e1_3, e2_3, e3_3;
    VectorReg<float> sum_0, sum_1, sum_2, sum_3;
    VectorReg<float8_e4m3_t> f0_0, f1_0, f2_0, f3_0, f0_1, f1_1, f2_1, f3_1,
        f0_2, f1_2, f2_2, f3_2, f0_3, f1_3, f2_3, f3_3;
    VectorReg<uint8_t> joined_0, part_0, reordered_0, joined_1, part_1,
        reordered_1, joined_2, part_2, reordered_2, joined_3, part_3,
        reordered_3;
    vector_align um, us;

    for (uint16_t r = 0; r < rows; r += 8) {
      const uint32_t soff = (r / 16) * sm + (r % 16) * 16;
      const uint32_t pof = (r / 16) * pm + (r % 16) * 32;

      // Fetch the running state before the column-wise maximum chain.
      vlds(previous, mi + r, 0, E2B_B16);
      // Four independent chains cover eight NZ16 blocks each.
      vlds(xmax_0, s + soff, 0, NORM);
      vlds(xmax_1, s + 8 * sn + soff, 0, NORM);
      vlds(xmax_2, s + 16 * sn + soff, 0, NORM);
      vlds(xmax_3, s + 24 * sn + soff, 0, NORM);
      for (uint16_t k = 1; k < 8; ++k) {
        vlds(x0_0, s + k * sn + soff, 0, NORM);
        vlds(x0_1, s + (8 + k) * sn + soff, 0, NORM);
        vlds(x0_2, s + (16 + k) * sn + soff, 0, NORM);
        vlds(x0_3, s + (24 + k) * sn + soff, 0, NORM);
        vmax(xmax_0, xmax_0, x0_0, halves, MODE_ZEROING);
        vmax(xmax_1, xmax_1, x0_1, halves, MODE_ZEROING);
        vmax(xmax_2, xmax_2, x0_2, halves, MODE_ZEROING);
        vmax(xmax_3, xmax_3, x0_3, halves, MODE_ZEROING);
      }
      vmax(xmax_0, xmax_0, xmax_1, halves, MODE_ZEROING);
      vmax(xmax_2, xmax_2, xmax_3, halves, MODE_ZEROING);
      vmax(xmax_0, xmax_0, xmax_2, halves, MODE_ZEROING);

      // Include the running maximum before reducing and broadcasting each row.
      vmax(xmax_0, xmax_0, previous, halves, MODE_ZEROING);
      vcgmax(packed_m, xmax_0, halves, MODE_ZEROING);
      vselr((VectorReg<uint16_t> &)mn, (VectorReg<uint16_t> &)packed_m, row);

      vdup(sum_0, 0.0f, floats, MODE_ZEROING);
      vdup(sum_1, 0.0f, floats, MODE_ZEROING);
      vdup(sum_2, 0.0f, floats, MODE_ZEROING);
      vdup(sum_3, 0.0f, floats, MODE_ZEROING);

      // Prefetch the first pairs before entering the exp/pack loop.
      vlds(x0_0, s + soff, 0, NORM);
      vlds(x1_0, s + sn + soff, 0, NORM);
      vlds(x0_1, s + 8 * sn + soff, 0, NORM);
      vlds(x1_1, s + 9 * sn + soff, 0, NORM);
      vlds(x0_2, s + 16 * sn + soff, 0, NORM);
      vlds(x1_2, s + 17 * sn + soff, 0, NORM);
      vlds(x0_3, s + 24 * sn + soff, 0, NORM);
      vlds(x1_3, s + 25 * sn + soff, 0, NORM);

      for (uint16_t k = 0; k < 3; ++k) {
        const uint16_t col0 = 2 * k;
        const uint16_t col1 = 8 + 2 * k;
        const uint16_t col2 = 16 + 2 * k;
        const uint16_t col3 = 24 + 2 * k;

        vexpdif(e0_0, x0_0, mn, halves, PART_EVEN);
        vexpdif(e0_1, x0_1, mn, halves, PART_EVEN);
        vexpdif(e0_2, x0_2, mn, halves, PART_EVEN);
        vexpdif(e0_3, x0_3, mn, halves, PART_EVEN);
        vexpdif(e2_0, x0_0, mn, halves, PART_ODD);
        vexpdif(e2_1, x0_1, mn, halves, PART_ODD);
        vexpdif(e2_2, x0_2, mn, halves, PART_ODD);
        vexpdif(e2_3, x0_3, mn, halves, PART_ODD);
        vexpdif(e1_0, x1_0, mn, halves, PART_EVEN);
        vexpdif(e1_1, x1_1, mn, halves, PART_EVEN);
        vexpdif(e1_2, x1_2, mn, halves, PART_EVEN);
        vexpdif(e1_3, x1_3, mn, halves, PART_EVEN);
        vexpdif(e3_0, x1_0, mn, halves, PART_ODD);
        vexpdif(e3_1, x1_1, mn, halves, PART_ODD);
        vexpdif(e3_2, x1_2, mn, halves, PART_ODD);
        vexpdif(e3_3, x1_3, mn, halves, PART_ODD);

        vadd(sum_0, sum_0, e0_0, floats, MODE_ZEROING);
        vadd(sum_1, sum_1, e0_1, floats, MODE_ZEROING);
        vadd(sum_2, sum_2, e0_2, floats, MODE_ZEROING);
        vadd(sum_3, sum_3, e0_3, floats, MODE_ZEROING);
        vadd(sum_0, sum_0, e2_0, floats, MODE_ZEROING);
        vadd(sum_1, sum_1, e2_1, floats, MODE_ZEROING);
        vadd(sum_2, sum_2, e2_2, floats, MODE_ZEROING);
        vadd(sum_3, sum_3, e2_3, floats, MODE_ZEROING);
        vadd(sum_0, sum_0, e1_0, floats, MODE_ZEROING);
        vadd(sum_1, sum_1, e1_1, floats, MODE_ZEROING);
        vadd(sum_2, sum_2, e1_2, floats, MODE_ZEROING);
        vadd(sum_3, sum_3, e1_3, floats, MODE_ZEROING);
        vadd(sum_0, sum_0, e3_0, floats, MODE_ZEROING);
        vadd(sum_1, sum_1, e3_1, floats, MODE_ZEROING);
        vadd(sum_2, sum_2, e3_2, floats, MODE_ZEROING);
        vadd(sum_3, sum_3, e3_3, floats, MODE_ZEROING);

        vcvt(f0_0, e0_0, floats, ROUND_R, RS_DISABLE, PART_P0, MODE_ZEROING);
        vcvt(f2_0, e2_0, floats, ROUND_R, RS_DISABLE, PART_P2, MODE_ZEROING);
        vcvt(f1_0, e1_0, floats, ROUND_R, RS_DISABLE, PART_P1, MODE_ZEROING);
        vcvt(f3_0, e3_0, floats, ROUND_R, RS_DISABLE, PART_P3, MODE_ZEROING);
        vor(joined_0, (VectorReg<uint8_t> &)f0_0, (VectorReg<uint8_t> &)f2_0,
            bytes, MODE_ZEROING);
        vor(part_0, (VectorReg<uint8_t> &)f1_0, (VectorReg<uint8_t> &)f3_0,
            bytes, MODE_ZEROING);
        vor(joined_0, joined_0, part_0, bytes, MODE_ZEROING);
        vselr(reordered_0, joined_0, indices);

        vcvt(f0_1, e0_1, floats, ROUND_R, RS_DISABLE, PART_P0, MODE_ZEROING);
        vcvt(f2_1, e2_1, floats, ROUND_R, RS_DISABLE, PART_P2, MODE_ZEROING);
        vcvt(f1_1, e1_1, floats, ROUND_R, RS_DISABLE, PART_P1, MODE_ZEROING);
        vcvt(f3_1, e3_1, floats, ROUND_R, RS_DISABLE, PART_P3, MODE_ZEROING);
        vor(joined_1, (VectorReg<uint8_t> &)f0_1, (VectorReg<uint8_t> &)f2_1,
            bytes, MODE_ZEROING);
        vor(part_1, (VectorReg<uint8_t> &)f1_1, (VectorReg<uint8_t> &)f3_1,
            bytes, MODE_ZEROING);
        vor(joined_1, joined_1, part_1, bytes, MODE_ZEROING);
        vselr(reordered_1, joined_1, indices);

        vcvt(f0_2, e0_2, floats, ROUND_R, RS_DISABLE, PART_P0, MODE_ZEROING);
        vcvt(f2_2, e2_2, floats, ROUND_R, RS_DISABLE, PART_P2, MODE_ZEROING);
        vcvt(f1_2, e1_2, floats, ROUND_R, RS_DISABLE, PART_P1, MODE_ZEROING);
        vcvt(f3_2, e3_2, floats, ROUND_R, RS_DISABLE, PART_P3, MODE_ZEROING);
        vor(joined_2, (VectorReg<uint8_t> &)f0_2, (VectorReg<uint8_t> &)f2_2,
            bytes, MODE_ZEROING);
        vor(part_2, (VectorReg<uint8_t> &)f1_2, (VectorReg<uint8_t> &)f3_2,
            bytes, MODE_ZEROING);
        vor(joined_2, joined_2, part_2, bytes, MODE_ZEROING);
        vselr(reordered_2, joined_2, indices);

        vcvt(f0_3, e0_3, floats, ROUND_R, RS_DISABLE, PART_P0, MODE_ZEROING);
        vcvt(f2_3, e2_3, floats, ROUND_R, RS_DISABLE, PART_P2, MODE_ZEROING);
        vcvt(f1_3, e1_3, floats, ROUND_R, RS_DISABLE, PART_P1, MODE_ZEROING);
        vcvt(f3_3, e3_3, floats, ROUND_R, RS_DISABLE, PART_P3, MODE_ZEROING);
        vor(joined_3, (VectorReg<uint8_t> &)f0_3, (VectorReg<uint8_t> &)f2_3,
            bytes, MODE_ZEROING);
        vor(part_3, (VectorReg<uint8_t> &)f1_3, (VectorReg<uint8_t> &)f3_3,
            bytes, MODE_ZEROING);
        vor(joined_3, joined_3, part_3, bytes, MODE_ZEROING);
        vselr(reordered_3, joined_3, indices);

        // All current scores are consumed. Reuse their registers for the
        // next pairs before storing P over the current even score blocks.
        vlds(x0_0, s + (col0 + 2) * sn + soff, 0, NORM);
        vlds(x1_0, s + (col0 + 3) * sn + soff, 0, NORM);
        vlds(x0_1, s + (col1 + 2) * sn + soff, 0, NORM);
        vlds(x1_1, s + (col1 + 3) * sn + soff, 0, NORM);
        vlds(x0_2, s + (col2 + 2) * sn + soff, 0, NORM);
        vlds(x1_2, s + (col2 + 3) * sn + soff, 0, NORM);
        vlds(x0_3, s + (col3 + 2) * sn + soff, 0, NORM);
        vlds(x1_3, s + (col3 + 3) * sn + soff, 0, NORM);

        vsts(reordered_0, po + (col0 / 2) * pn + pof, 0, NORM_B8, bytes);
        vsts(reordered_1, po + (col1 / 2) * pn + pof, 0, NORM_B8, bytes);
        vsts(reordered_2, po + (col2 / 2) * pn + pof, 0, NORM_B8, bytes);
        vsts(reordered_3, po + (col3 / 2) * pn + pof, 0, NORM_B8, bytes);
      }

      {
        const uint16_t col0 = 6;
        const uint16_t col1 = 14;
        const uint16_t col2 = 22;
        const uint16_t col3 = 30;

        vexpdif(e0_0, x0_0, mn, halves, PART_EVEN);
        vexpdif(e0_1, x0_1, mn, halves, PART_EVEN);
        vexpdif(e0_2, x0_2, mn, halves, PART_EVEN);
        vexpdif(e0_3, x0_3, mn, halves, PART_EVEN);
        vexpdif(e2_0, x0_0, mn, halves, PART_ODD);
        vexpdif(e2_1, x0_1, mn, halves, PART_ODD);
        vexpdif(e2_2, x0_2, mn, halves, PART_ODD);
        vexpdif(e2_3, x0_3, mn, halves, PART_ODD);
        vexpdif(e1_0, x1_0, mn, halves, PART_EVEN);
        vexpdif(e1_1, x1_1, mn, halves, PART_EVEN);
        vexpdif(e1_2, x1_2, mn, halves, PART_EVEN);
        vexpdif(e1_3, x1_3, mn, halves, PART_EVEN);
        vexpdif(e3_0, x1_0, mn, halves, PART_ODD);
        vexpdif(e3_1, x1_1, mn, halves, PART_ODD);
        vexpdif(e3_2, x1_2, mn, halves, PART_ODD);
        vexpdif(e3_3, x1_3, mn, halves, PART_ODD);

        vadd(sum_0, sum_0, e0_0, floats, MODE_ZEROING);
        vadd(sum_1, sum_1, e0_1, floats, MODE_ZEROING);
        vadd(sum_2, sum_2, e0_2, floats, MODE_ZEROING);
        vadd(sum_3, sum_3, e0_3, floats, MODE_ZEROING);
        vadd(sum_0, sum_0, e2_0, floats, MODE_ZEROING);
        vadd(sum_1, sum_1, e2_1, floats, MODE_ZEROING);
        vadd(sum_2, sum_2, e2_2, floats, MODE_ZEROING);
        vadd(sum_3, sum_3, e2_3, floats, MODE_ZEROING);
        vadd(sum_0, sum_0, e1_0, floats, MODE_ZEROING);
        vadd(sum_1, sum_1, e1_1, floats, MODE_ZEROING);
        vadd(sum_2, sum_2, e1_2, floats, MODE_ZEROING);
        vadd(sum_3, sum_3, e1_3, floats, MODE_ZEROING);
        vadd(sum_0, sum_0, e3_0, floats, MODE_ZEROING);
        vadd(sum_1, sum_1, e3_1, floats, MODE_ZEROING);
        vadd(sum_2, sum_2, e3_2, floats, MODE_ZEROING);
        vadd(sum_3, sum_3, e3_3, floats, MODE_ZEROING);

        vcvt(f0_0, e0_0, floats, ROUND_R, RS_DISABLE, PART_P0, MODE_ZEROING);
        vcvt(f2_0, e2_0, floats, ROUND_R, RS_DISABLE, PART_P2, MODE_ZEROING);
        vcvt(f1_0, e1_0, floats, ROUND_R, RS_DISABLE, PART_P1, MODE_ZEROING);
        vcvt(f3_0, e3_0, floats, ROUND_R, RS_DISABLE, PART_P3, MODE_ZEROING);
        vor(joined_0, (VectorReg<uint8_t> &)f0_0, (VectorReg<uint8_t> &)f2_0,
            bytes, MODE_ZEROING);
        vor(part_0, (VectorReg<uint8_t> &)f1_0, (VectorReg<uint8_t> &)f3_0,
            bytes, MODE_ZEROING);
        vor(joined_0, joined_0, part_0, bytes, MODE_ZEROING);
        vselr(reordered_0, joined_0, indices);

        vcvt(f0_1, e0_1, floats, ROUND_R, RS_DISABLE, PART_P0, MODE_ZEROING);
        vcvt(f2_1, e2_1, floats, ROUND_R, RS_DISABLE, PART_P2, MODE_ZEROING);
        vcvt(f1_1, e1_1, floats, ROUND_R, RS_DISABLE, PART_P1, MODE_ZEROING);
        vcvt(f3_1, e3_1, floats, ROUND_R, RS_DISABLE, PART_P3, MODE_ZEROING);
        vor(joined_1, (VectorReg<uint8_t> &)f0_1, (VectorReg<uint8_t> &)f2_1,
            bytes, MODE_ZEROING);
        vor(part_1, (VectorReg<uint8_t> &)f1_1, (VectorReg<uint8_t> &)f3_1,
            bytes, MODE_ZEROING);
        vor(joined_1, joined_1, part_1, bytes, MODE_ZEROING);
        vselr(reordered_1, joined_1, indices);

        vcvt(f0_2, e0_2, floats, ROUND_R, RS_DISABLE, PART_P0, MODE_ZEROING);
        vcvt(f2_2, e2_2, floats, ROUND_R, RS_DISABLE, PART_P2, MODE_ZEROING);
        vcvt(f1_2, e1_2, floats, ROUND_R, RS_DISABLE, PART_P1, MODE_ZEROING);
        vcvt(f3_2, e3_2, floats, ROUND_R, RS_DISABLE, PART_P3, MODE_ZEROING);
        vor(joined_2, (VectorReg<uint8_t> &)f0_2, (VectorReg<uint8_t> &)f2_2,
            bytes, MODE_ZEROING);
        vor(part_2, (VectorReg<uint8_t> &)f1_2, (VectorReg<uint8_t> &)f3_2,
            bytes, MODE_ZEROING);
        vor(joined_2, joined_2, part_2, bytes, MODE_ZEROING);
        vselr(reordered_2, joined_2, indices);

        vcvt(f0_3, e0_3, floats, ROUND_R, RS_DISABLE, PART_P0, MODE_ZEROING);
        vcvt(f2_3, e2_3, floats, ROUND_R, RS_DISABLE, PART_P2, MODE_ZEROING);
        vcvt(f1_3, e1_3, floats, ROUND_R, RS_DISABLE, PART_P1, MODE_ZEROING);
        vcvt(f3_3, e3_3, floats, ROUND_R, RS_DISABLE, PART_P3, MODE_ZEROING);
        vor(joined_3, (VectorReg<uint8_t> &)f0_3, (VectorReg<uint8_t> &)f2_3,
            bytes, MODE_ZEROING);
        vor(part_3, (VectorReg<uint8_t> &)f1_3, (VectorReg<uint8_t> &)f3_3,
            bytes, MODE_ZEROING);
        vor(joined_3, joined_3, part_3, bytes, MODE_ZEROING);
        vselr(reordered_3, joined_3, indices);

        vsts(reordered_0, po + (col0 / 2) * pn + pof, 0, NORM_B8, bytes);
        vsts(reordered_1, po + (col1 / 2) * pn + pof, 0, NORM_B8, bytes);
        vsts(reordered_2, po + (col2 / 2) * pn + pof, 0, NORM_B8, bytes);
        vsts(reordered_3, po + (col3 / 2) * pn + pof, 0, NORM_B8, bytes);
      }

      // Merge the four FP32 partial sums before reducing each row.
      vadd(sum_0, sum_0, sum_1, floats, MODE_ZEROING);
      vadd(sum_2, sum_2, sum_3, floats, MODE_ZEROING);
      vadd(sum_0, sum_0, sum_2, floats, MODE_ZEROING);
      vcgadd(sum_0, sum_0, floats, MODE_ZEROING);
      // Store the state after the score loads to avoid a store/load dependency.
      vstus(um, 8, packed_m, mo, POST_UPDATE);
      vstus(us, 8, (VectorReg<int32_t> &)sum_0, (__ubuf__ int32_t *&)so,
            POST_UPDATE); // block_sum
    }
    vstas(um, mo, 0, POST_UPDATE);
    vstas(us, (__ubuf__ int32_t *&)so, 0, POST_UPDATE);
  }
}

// Mask the existing FP16 score allocation once before the joint N512 softmax.
// Keeping this VF separate avoids extending the four exp chains' register
// lifetimes with masking temporaries or repeating comparisons on score reloads.
__aicore__ __attribute__((always_inline)) void
maskOnlineSoftmaxF16(memref_t<__ubuf__ half, 4> *scores,
                     memref_t<__ubuf__ int32_t, 1> *causal_limits) {
  auto s = scores->aligned + scores->offset;
  auto cl = causal_limits->aligned + causal_limits->offset;
  const uint16_t rows = scores->sizes[1] * 16;
  const uint32_t sn = scores->strides[0], sm = scores->strides[1];
  __VEC_SCOPE__ {
    vector_bool halves = pset_b16(PAT_ALL);
    vector_bool ints = pset_b32(PAT_ALL);
    VectorReg<uint16_t> lane, row, columns, limitIndices, fifteen;
    VectorReg<int32_t> limits, lower, upper;
    VectorReg<int16_t> rowLimits;
    VectorReg<half> negativeInf, score;
    vci((VectorReg<int16_t> &)lane, int16_t(0), INC_ORDER);
    vshrs(row, lane, int16_t(4), halves, MODE_ZEROING);
    vdup(fifteen, uint16_t(15), halves, MODE_ZEROING);
    vand(columns, lane, fifteen, halves, MODE_ZEROING);
    vshls(limitIndices, row, int16_t(4), halves, MODE_ZEROING);
    vdup(negativeInf, half(-__builtin_inff()), halves, MODE_ZEROING);
    vdup(lower, int32_t(-1), ints, MODE_ZEROING);
    vdup(upper, int32_t(511), ints, MODE_ZEROING);

    for (uint16_t r = 0; r < rows; r += 8) {
      const uint32_t soff = (r / 16) * sm + (r % 16) * 16;
      vlds(limits, cl + r, 0, E2B_B32);
      vmax(limits, limits, lower, ints, MODE_ZEROING);
      vmin(limits, limits, upper, ints, MODE_ZEROING);
      // E2B repeats each int32 limit eight times. Broadcast its low
      // halfword at 16*row to the corresponding 16 FP16 score lanes.
      vselr((VectorReg<uint16_t> &)rowLimits, (VectorReg<uint16_t> &)limits,
            limitIndices);
      for (uint16_t block = 0; block < 32; ++block) {
        VectorReg<uint16_t> absoluteColumns;
        vadds(absoluteColumns, columns, uint16_t(block * 16), halves,
              MODE_ZEROING);
        vector_bool visible;
        vcmp_le(visible, (VectorReg<int16_t> &)absoluteColumns, rowLimits,
                halves);
        vlds(score, s + block * sn + soff, 0, NORM);
        vsel((VectorReg<uint16_t> &)score, (VectorReg<uint16_t> &)score,
             (VectorReg<uint16_t> &)negativeInf, visible);
        vsts(score, s + block * sn + soff, 0, NORM_B16, halves);
      }
    }
    // The following softmax VF reloads the scores written by this VF.
    // PIPE_V barriers are no-ops on dav_3510, so use the UB store/load barrier.
    mem_bar(VST_VLD);
  }
}

extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_online_softmax_nz_f16_f8e4m3fn_causal(
    memref_t<__ubuf__ half, 4> *scores, memref_t<__ubuf__ half, 1> *old_m,
    memref_t<__ubuf__ uint8_t, 1> *pack_index,
    memref_t<__ubuf__ int32_t, 1> *causal_limits,
    memref_t<__ubuf__ float8_e4m3_t, 4> *p, memref_t<__ubuf__ half, 1> *new_m,
    memref_t<__ubuf__ float, 1> *block_sum) {
  maskOnlineSoftmaxF16(scores, causal_limits);
  _mlir_ciface_online_softmax_nz_f16_f8e4m3fn(scores, old_m, pack_index, p,
                                              new_m, block_sum);
}

// The caller supplies whether any score row requires masking, avoiding UB
// scalar reads and an additional V_S/S_V synchronization pair for history
// tiles.
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_online_softmax_nz_f16_f8e4m3fn_causal_if(
    memref_t<__ubuf__ half, 4> *scores, memref_t<__ubuf__ half, 1> *old_m,
    memref_t<__ubuf__ uint8_t, 1> *pack_index,
    memref_t<__ubuf__ int32_t, 1> *causal_limits, bool mask_required,
    memref_t<__ubuf__ float8_e4m3_t, 4> *p, memref_t<__ubuf__ half, 1> *new_m,
    memref_t<__ubuf__ float, 1> *block_sum) {
  if (mask_required)
    maskOnlineSoftmaxF16(scores, causal_limits);
  _mlir_ciface_online_softmax_nz_f16_f8e4m3fn(scores, old_m, pack_index, p,
                                              new_m, block_sum);
}

#endif // BISHENG_TPL_C310
